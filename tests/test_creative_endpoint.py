import time
from decimal import ROUND_HALF_UP, Decimal
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from core.errors import CONTENT_FILTERED_MESSAGE
from core.placeholders import PlaceholdersRemainError
from db.models import Asset
from services import creative, higgsfield, label_check, storage
from tests.test_asset_model import make_asset
from tests.test_creative import make_item, make_plan

PAYLOAD = {
    "brief": "Spring sale on trail runners",
    "brand": {
        "brand_name": "Acme", "voice": "playful", "colors": ["red", "white"],
        "offer": "trail running shoes", "audience": "weekend runners",
    },
    "goal": "sales",
    "platform": "instagram",
}


@pytest.fixture
def mocks(monkeypatch):
    """OpenAI, Higgsfield and storage all faked. Set `fail` / `upload_fail` to a prompt -> exception."""
    state = SimpleNamespace(
        planner_calls=[], generated=[], uploads=[], fail={}, upload_fail={},
        active_uploads=0, max_parallel_uploads=0, plan=None,
    )

    def create_with_completion(**kwargs):
        state.planner_calls.append(kwargs)
        # 3000 in / 1000 out tokens = 0.00045 + 0.0006 = $0.00105 planning cost
        usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=1000, total_tokens=4000)
        # Validate like Instructor does, with the request's spec as context.
        plan = state.plan or make_plan()
        return creative.CreativePlan.model_validate(plan.model_dump(), context=kwargs["context"]), SimpleNamespace(usage=usage)

    def generate_image(prompt):
        key = key_of(prompt)
        state.generated.append(key)
        if key in state.fail:
            raise state.fail[key]
        return higgsfield.GeneratedImage(data=f"img:{key}".encode(), content_type="image/png", request_id=f"r-{key}")

    def upload_asset(data, content_type, extension=""):
        state.active_uploads += 1
        state.max_parallel_uploads = max(state.max_parallel_uploads, state.active_uploads)
        time.sleep(0.01)  # give any parallel upload a chance to overlap
        state.active_uploads -= 1
        state.uploads.append((data, content_type, extension))
        prompt = data.decode().removeprefix("img:")
        if prompt in state.upload_fail:
            raise state.upload_fail[prompt]
        return f"https://cdn.example/{prompt}.{extension}"

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion)))
    monkeypatch.setattr(creative, "client", fake_client)
    monkeypatch.setattr(higgsfield, "generate_image", generate_image)
    monkeypatch.setattr(storage, "upload_asset", upload_asset)
    return state


def key_of(prompt):
    """make_item prompts end with a short key ("track", "beach", "city"); the router appends its rules after it."""
    for rule in (creative.REALISM_PROMPT, creative.SAFER_REALISM_PROMPT, creative.ONE_PRODUCT,
                 creative.LABEL_FACING_RULE, creative.IN_USE_LABEL_RULE, creative.IN_USE_LABEL_RULE_NO_PEOPLE,
                 " The product label"):
        prompt = prompt.split(" " + rule.strip())[0]
    return prompt.split()[-1]


def saved_assets(engine):
    with Session(engine) as session:
        return session.exec(select(Asset).order_by(Asset.id)).all()


def test_generate_saves_three_drafts_with_format_and_real_cost(client, engine, mocks):
    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 200, resp.text
    body = resp.json()["assets"]
    assert [a["format"] for a in body] == ["ugc", "lifestyle", "flat_lay"]
    assert [a["status"] for a in body] == ["draft"] * 3
    assert [a["file_url"] for a in body] == [f"https://cdn.example/{p}.png" for p in ("track", "beach", "city")]
    assert [a["needs_input"] for a in body] == [[], [], []]
    assert resp.json()["failed_count"] == 0 and resp.json()["failed"] == []

    assets = saved_assets(engine)
    assert len(assets) == 3
    for asset in assets:
        assert asset.campaign_brief == PAYLOAD["brief"]
        assert asset.kind == "image"
        assert asset.provider == "higgsfield"
        assert asset.model == higgsfield.MODEL
        assert len(asset.faq) == 2
        assert asset.offer_type == "physical_product"
        # $0.006 image + $0.00105 / 3 planning = $0.00635, rounded to 4 places
        assert asset.cost_usd == Decimal("0.0064")
    assert sorted(mocks.generated) == ["beach", "city", "track"]


def test_response_lists_placeholders_still_needing_input(client, monkeypatch, mocks):
    plan = make_plan()
    plan.items[0].caption = "Free returns within [▢ return window, e.g. 30 days]."
    monkeypatch.setattr(creative, "plan_creatives", lambda request: (plan, Decimal("0.001")))

    body = client.post("/creative/generate", json=PAYLOAD).json()["assets"]

    assert body[0]["needs_input"] == ["[▢ return window, e.g. 30 days]"]
    assert body[1]["needs_input"] == []


def test_uploads_run_one_at_a_time_in_plan_order(client, mocks):
    client.post("/creative/generate", json=PAYLOAD)

    assert mocks.max_parallel_uploads == 1
    assert [u[0] for u in mocks.uploads] == [b"img:track", b"img:beach", b"img:city"]


def test_real_build_prompt_reaches_the_planner(client, mocks):
    client.post("/creative/generate", json=PAYLOAD)

    content = mocks.planner_calls[0]["messages"][0]["content"]
    assert PAYLOAD["brief"] in content
    assert "Acme" in content and "red, white" in content
    assert "sales" in content and "instagram" in content


def test_spend_cap_blocks_before_any_spend(client, engine, mocks):
    with Session(engine) as session:
        session.add(make_asset(cost_usd=Decimal("4.99")))  # 4.99 + ~0.028 estimate > 5.00
        session.commit()

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 402
    assert resp.json()["code"] == "spend_cap_reached"
    assert mocks.planner_calls == [] and mocks.generated == [] and mocks.uploads == []
    assert len(saved_assets(engine)) == 1


# --- physical_product runs with a product photo ---

PHOTO = "f51db36f3c08442daae2a6325a563aac.jpg"
SIGNED = f"https://proj.supabase.co/storage/v1/object/sign/product-photos/{PHOTO}?token=SECRET-TOKEN"
LABEL = (
    " The product label reads exactly «Acme» with the apostrophe. Keep all label text exactly as in the "
    "reference photo; do not change or invent any text. Do not add objects beyond those described in this prompt."
)
REFERENCE = "ACMF\nTRAIL RUNNER\nsize 9"  # as read off the photo: brand misread, corrected to "Acme" before checks
CORRECTED = "Acme TRAIL RUNNER size 9"


@pytest.fixture
def photo_mocks(monkeypatch, mocks):
    """Adds the signed URL, Marketing Studio and the label check to `mocks`.

    Records (prompt, reference_url) and label-check calls. Put an image key in `label_bad` to fail its check.
    `saved_reference` is the label text stored for the photo (None: never read); reads of the photo
    itself return `photo_label` and are recorded in `photo_reads`, saves in `saved_references`.
    """
    mocks.product_calls, mocks.signed, mocks.label_checks, mocks.label_bad = [], [], [], set()
    mocks.saved_reference, mocks.photo_label = REFERENCE, REFERENCE
    mocks.photo_reads, mocks.saved_references, mocks.brand_required = [], [], {}

    def check_label(image, content_type, brand_name, reference=None, brand_required=False):
        key = image.decode().removeprefix("img:")
        mocks.label_checks.append((key, brand_name, reference))
        mocks.brand_required[key] = brand_required
        if key in mocks.label_bad:
            return label_check.LabelCheck(ok=False, text_read="TADUNE", cost_usd=Decimal("0.004"))
        return label_check.LabelCheck(ok=True, text_read="J'ADUNÉ", cost_usd=Decimal("0.004"))

    monkeypatch.setattr(label_check, "check_label", check_label)

    def create_signed_url(bucket, path):
        mocks.signed.append((bucket, path))
        return SIGNED

    def generate_product_image(prompt, reference_url):
        mocks.product_calls.append((prompt, reference_url))
        key = key_of(prompt)
        return higgsfield.GeneratedImage(data=f"img:{key}".encode(), content_type="image/png", request_id=f"r-{key}")

    def read_label(image, content_type):
        mocks.photo_reads.append(image)
        return mocks.photo_label, Decimal("0.003")

    monkeypatch.setattr(storage, "create_signed_url", create_signed_url)
    monkeypatch.setattr(storage, "download_label_reference", lambda path: mocks.saved_reference)
    monkeypatch.setattr(storage, "upload_label_reference", lambda path, text: mocks.saved_references.append((path, text)))
    monkeypatch.setattr(storage, "download_product_photo", lambda path: b"photo-bytes")
    monkeypatch.setattr(higgsfield, "generate_product_image", generate_product_image)
    monkeypatch.setattr(label_check, "read_label", read_label)
    return mocks


def test_photo_run_uses_marketing_studio_with_the_signed_url_and_label(client, engine, photo_mocks):
    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 200, resp.text
    assert photo_mocks.signed == [("product-photos", PHOTO)]
    assert photo_mocks.generated == []  # Soul v2 not used
    assert len(photo_mocks.product_calls) == 3
    assert {url for _, url in photo_mocks.product_calls} == {SIGNED}
    assert all(prompt.endswith(LABEL) for prompt, _ in photo_mocks.product_calls)

    assets = saved_assets(engine)
    sent = {prompt for prompt, _ in photo_mocks.product_calls}
    assert {asset.prompt for asset in assets} == sent  # saved exactly as sent
    for asset in assets:
        assert asset.model == "marketing-studio/image"
        assert asset.provider == "higgsfield"
        # the image priced from the prompt sent + $0.004 label check + $0.00105 / 3 planning
        expected = higgsfield.product_image_cost(asset.prompt) + Decimal("0.004") + Decimal("0.00035")
        assert asset.cost_usd == expected.quantize(Decimal("0.0001"), ROUND_HALF_UP)


def test_each_format_gets_its_own_label_rule(client, photo_mocks):
    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    in_use = (creative.IN_USE_LABEL_RULE, creative.IN_USE_LABEL_RULE_NO_PEOPLE)
    by_key = {key_of(prompt): (creative.LABEL_FACING_RULE in prompt, any(rule in prompt for rule in in_use))
              for prompt, _ in photo_mocks.product_calls}
    # ugc and lifestyle are shown in use; flat_lay gets neither rule
    assert by_key == {"track": (False, True), "beach": (False, True), "city": (False, False)}


def test_every_photo_image_gets_a_label_check_against_brand_name(client, engine, photo_mocks):
    photo_mocks.label_bad = {"beach"}

    body = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO}).json()["assets"]

    assert sorted(photo_mocks.label_checks) == [
        ("beach", "Acme", CORRECTED), ("city", "Acme", CORRECTED), ("track", "Acme", CORRECTED),
    ]
    assert [a["needs_input"] for a in body] == [[], ["label_mismatch"], []]
    assert [a["label_ok"] for a in body] == [True, False, True]
    assert body[1]["label_read"] == "TADUNE"
    assert [(a.label_ok, a.label_read) for a in saved_assets(engine)] == [
        (True, "J'ADUNÉ"), (False, "TADUNE"), (True, "J'ADUNÉ"),
    ]


def test_both_reads_are_logged_on_the_asset(client, engine, monkeypatch, photo_mocks):
    def check_label(image, content_type, brand_name, reference=None, brand_required=False):
        return label_check.LabelCheck(ok=True, text_read="JADUNÉ", cost_usd=Decimal("0.008"),
                                       read_twice=True, second_read="J'ADUNÉ")

    monkeypatch.setattr(label_check, "check_label", check_label)

    body = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO}).json()["assets"]

    logged = "JADUNÉ\n--- second read ---\nJ'ADUNÉ"
    assert [(a["label_ok"], a["label_read"]) for a in body] == [(True, logged)] * 3
    assert {a.label_read for a in saved_assets(engine)} == {logged}


def test_label_outcome_is_saved_on_every_asset(client, engine, monkeypatch, photo_mocks):
    checks = {
        "track": label_check.LabelCheck(ok=True, text_read="J'ADUNÉ", cost_usd=Decimal("0.008")),
        "beach": label_check.LabelCheck(ok=True, text_read="J'ADUNÉ [unreadable]", cost_usd=Decimal("0.008"),
                                        outcome=label_check.UNREADABLE),
        "city": label_check.LabelCheck(ok=False, text_read="J'ADUNE", cost_usd=Decimal("0.012"),
                                       read_twice=True, second_read="J'ADUNE"),
    }
    monkeypatch.setattr(label_check, "check_label",
                        lambda image, content_type, brand_name, reference=None, brand_required=False: checks[image.decode().removeprefix("img:")])

    body = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO}).json()["assets"]

    expected = [(True, "match"), (True, "unreadable"), (False, "mismatch")]
    assert [(a["label_ok"], a["label_outcome"]) for a in body] == expected
    assert [(a.label_ok, a.label_outcome) for a in saved_assets(engine)] == expected


def test_a_check_where_every_read_failed_saves_no_outcome(client, engine, monkeypatch, photo_mocks):
    failed = label_check.LabelCheck(ok=False, text_read=None, cost_usd=Decimal("0.008"), outcome=label_check.FAILED)
    monkeypatch.setattr(label_check, "check_label", lambda *args, **kwargs: failed)

    body = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO}).json()["assets"]

    # The column's CHECK allows only match / unreadable / mismatch, so a failed check is NULL with label_ok False.
    assert [(a.label_ok, a.label_outcome) for a in saved_assets(engine)] == [(False, None)] * 3
    assert [a["needs_input"] for a in body] == [["label_mismatch"]] * 3


def test_only_label_facing_formats_require_the_brand(client, photo_mocks):
    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    # ugc, lifestyle, flat_lay: the brand may be covered in all three
    assert photo_mocks.brand_required == {"track": False, "beach": False, "city": False}


def test_studio_hero_and_model_with_product_require_the_brand(client, mocks, photo_mocks):
    mocks.plan = angle_plan(["moms", "gifting"], ["model_with_product", "studio_hero"])

    resp = client.post("/creative/generate", json={**PAYLOAD, "angles": ["moms", "gifting"], "product_photo_path": PHOTO})

    assert resp.status_code == 200, resp.text
    assert photo_mocks.brand_required == {"a0": True, "a1": True}


def test_label_mismatch_blocks_approval(client, engine, photo_mocks):
    photo_mocks.label_bad = {"track"}
    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    with Session(engine) as session:
        flagged, fine, _ = session.exec(select(Asset).order_by(Asset.id)).all()
        fine.status = "approved"
        session.commit()
        flagged.status = "approved"
        with pytest.raises(PlaceholdersRemainError) as info:
            session.commit()
    assert info.value.placeholders == ["label_mismatch"]


def test_runs_without_a_photo_skip_the_label_check(client, engine, photo_mocks):
    body = client.post("/creative/generate", json=PAYLOAD).json()["assets"]

    assert photo_mocks.label_checks == []
    assert [a["label_ok"] for a in body] == [None, None, None]
    assert [a["label_outcome"] for a in body] == [None, None, None]
    assert [a["needs_input"] for a in body] == [[], [], []]


def test_signed_url_is_never_returned_or_saved(client, engine, photo_mocks):
    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert "SECRET-TOKEN" not in resp.text
    assert not any("SECRET-TOKEN" in asset.model_dump_json() for asset in saved_assets(engine))


def test_run_without_a_photo_stays_on_soul_with_no_label_text(client, engine, photo_mocks):
    assert client.post("/creative/generate", json=PAYLOAD).status_code == 200

    assert photo_mocks.signed == [] and photo_mocks.product_calls == []
    assert sorted(photo_mocks.generated) == ["beach", "city", "track"]
    for asset in saved_assets(engine):
        assert asset.model == higgsfield.MODEL
        assert "product label" not in asset.prompt


def test_spend_estimate_uses_the_higher_price_only_with_a_photo(client, engine, photo_mocks):
    with Session(engine) as session:
        # + $0.21 with a photo (3 x ($0.0456 image + $0.015 check) + reserves) > 5.00; + $0.028 without
        session.add(make_asset(cost_usd=Decimal("4.92")))
        session.commit()

    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 402
    assert photo_mocks.signed == [] and photo_mocks.planner_calls == [] and photo_mocks.product_calls == []
    assert client.post("/creative/generate", json=PAYLOAD).status_code == 200


def test_unusually_long_prompts_are_rechecked_against_the_cap_before_generating(client, engine, photo_mocks):
    long = make_plan()
    for item in long.items:
        item.prompt += " soft morning light" * 400  # ~7200 more characters each: well above the reserve
    photo_mocks.plan = long
    with Session(engine) as session:
        # The reserve (3 x ($0.0456 + 3 x $0.005) + $0.015 = $0.197) fits under $5.00; these prompts don't.
        session.add(make_asset(cost_usd=Decimal("4.80")))
        session.commit()

    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 402
    assert len(photo_mocks.planner_calls) == 1  # passed the first check; blocked once the prompts were known
    assert photo_mocks.product_calls == []  # nothing generated


def test_missing_photo_is_400_before_anything_is_spent(client, monkeypatch, photo_mocks):
    def not_found(bucket, path):
        raise storage.StorageObjectNotFound("Could not sign the file URL (status 400).")

    monkeypatch.setattr(storage, "create_signed_url", not_found)

    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_product_photo"
    assert photo_mocks.planner_calls == [] and photo_mocks.product_calls == []


@pytest.mark.parametrize("path", [
    "../assets/abc.jpg", "BE9A9FEE-8F9B-49C7-90D4-3A9D33CB3837_4_5005_c.jpeg", "abc.jpg", f"{PHOTO}\n", "",
])
def test_only_paths_from_the_upload_endpoint_are_accepted(client, photo_mocks, path):
    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": path})

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_input"
    assert photo_mocks.signed == [] and photo_mocks.planner_calls == []


@pytest.mark.parametrize("offer_type", ["service", "content"])
def test_photo_is_rejected_for_other_offer_types(client, photo_mocks, offer_type):
    resp = client.post("/creative/generate", json={**PAYLOAD, "offer_type": offer_type, "product_photo_path": PHOTO})

    assert resp.status_code == 400
    assert "physical_product" in resp.text
    assert photo_mocks.signed == [] and photo_mocks.planner_calls == []


def test_product_size_reaches_every_image_prompt(client, photo_mocks):
    size = "8 fl oz spray bottle, about 7 in tall"
    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO, "product_size": f"  {size} "})

    assert resp.status_code == 200, resp.text
    assert all(f"The product ({size}) is shown at true real-world size" in p for p, _ in photo_mocks.product_calls)


@pytest.mark.parametrize("size, status", [("", 200), ("x" * 101, 400)])
def test_blank_product_size_is_ignored_and_long_one_rejected(client, photo_mocks, size, status):
    resp = client.post("/creative/generate", json={**PAYLOAD, "product_size": size})

    assert resp.status_code == status, resp.text
    if status == 200:
        assert all(creative.SCALE_DEFAULT in a["prompt"] for a in resp.json()["assets"])


def test_product_size_is_rejected_for_other_offer_types(client, photo_mocks):
    resp = client.post("/creative/generate", json={**PAYLOAD, "offer_type": "service", "product_size": "7 in"})

    assert resp.status_code == 400
    assert "product_size is only used" in resp.text
    assert photo_mocks.planner_calls == []


@pytest.mark.parametrize("error, code", [
    (higgsfield.HiggsfieldTimeout("Image generation did not finish within 120s."), "image_timeout"),
    (higgsfield.HiggsfieldError("Image generation ended as 'failed'."), "image_provider_error"),
])
def test_one_failed_image_returns_200_with_drafts_and_failed_list(client, engine, mocks, error, code):
    mocks.fail["beach"] = error

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [a["format"] for a in body["assets"]] == ["ugc", "flat_lay"]
    assert body["failed_count"] == 1
    assert body["failed"] == [{"format": "lifestyle", "code": code, "reason": str(error), "asset_id": None}]
    assets = saved_assets(engine)
    assert sorted(a.format for a in assets) == ["flat_lay", "ugc"]
    # Planning cost is spread over the 2 saved assets: 0.006 + 0.00105 / 2 = 0.006525
    assert all(a.cost_usd == Decimal("0.0065") for a in assets)


FILTERED = higgsfield.HiggsfieldContentFiltered("Image generation ended as 'nsfw'.")


def test_content_filtered_image_is_retried_once_with_safer_wording(client, engine, mocks, monkeypatch):
    sent = []
    real = higgsfield.generate_image

    def generate_image(prompt):
        sent.append(prompt)
        if key_of(prompt) == "track" and creative.SAFER_REALISM_PROMPT not in prompt:
            raise FILTERED
        return real(prompt)

    monkeypatch.setattr(higgsfield, "generate_image", generate_image)

    body = client.post("/creative/generate", json=PAYLOAD).json()

    assert body["failed"] == [] and len(body["assets"]) == 3
    track = [p for p in sent if key_of(p) == "track"]
    assert len(track) == 2 and creative.REALISM_PROMPT in track[0] and creative.SAFER_REALISM_PROMPT in track[1]
    ugc = next(a for a in saved_assets(engine) if a.format == "ugc")
    assert ugc.prompt == track[1]  # saved as actually sent
    assert ugc.cost_usd == Decimal("0.0064")  # one image: the rejection wasn't charged


def test_image_filtered_twice_is_content_filtered_with_a_friendly_message(client, engine, mocks):
    mocks.fail["beach"] = FILTERED

    body = client.post("/creative/generate", json=PAYLOAD).json()

    assert mocks.generated.count("beach") == 2  # one retry, no more
    assert body["failed"] == [{
        "format": "lifestyle", "code": "content_filtered", "reason": CONTENT_FILTERED_MESSAGE, "asset_id": None,
    }]
    assert "weren't charged" in CONTENT_FILTERED_MESSAGE
    assert len(saved_assets(engine)) == 2


def test_every_image_filtered_is_422_content_filtered(client, engine, mocks):
    mocks.fail.update({"track": FILTERED, "beach": FILTERED, "city": FILTERED})

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 422
    assert resp.json()["code"] == "content_filtered"
    assert resp.json()["detail"] == CONTENT_FILTERED_MESSAGE
    assert len(mocks.generated) == 6 and saved_assets(engine) == []


def test_other_provider_failures_are_not_retried(client, mocks):
    mocks.fail["beach"] = higgsfield.HiggsfieldError("Image generation ended as 'failed'.")

    client.post("/creative/generate", json=PAYLOAD)

    assert mocks.generated.count("beach") == 1


def test_failed_upload_returns_200_and_points_at_the_saved_paid_asset(client, engine, mocks):
    # The live case: one upload hit a network error (after storage's own retries).
    mocks.upload_fail["city"] = storage.StorageError("Storage upload failed: network error (ReadError).")

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert [a["format"] for a in body["assets"]] == ["ugc", "lifestyle"]
    assert all(a["file_url"] for a in body["assets"])
    assets = {a.format: a for a in saved_assets(engine)}
    assert len(assets) == 3  # the paid image is still recorded for the spend cap
    assert assets["flat_lay"].file_url is None
    assert body["failed_count"] == 1
    assert body["failed"] == [{
        "format": "flat_lay", "code": "storage_error",
        "reason": "Storage upload failed: network error (ReadError).",
        "asset_id": assets["flat_lay"].id,
    }]


def test_mixed_failures_are_all_listed(client, mocks):
    mocks.fail["track"] = higgsfield.HiggsfieldTimeout("slow")
    mocks.upload_fail["city"] = storage.StorageError("Storage upload failed with status 503.")

    body = client.post("/creative/generate", json=PAYLOAD).json()

    assert [a["format"] for a in body["assets"]] == ["lifestyle"]
    assert body["failed_count"] == 2
    assert [(f["format"], f["code"]) for f in body["failed"]] == [("ugc", "image_timeout"), ("flat_lay", "storage_error")]


def test_all_generations_failing_is_502_and_saves_nothing(client, engine, mocks):
    mocks.fail.update({p: higgsfield.HiggsfieldError("failed") for p in ("track", "beach", "city")})

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 502
    assert resp.json()["code"] == "image_provider_error"
    assert saved_assets(engine) == []


def test_all_uploads_failing_is_502_but_keeps_paid_assets(client, engine, mocks):
    mocks.upload_fail.update({p: storage.StorageError("Storage upload failed.") for p in ("track", "beach", "city")})

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 502
    assert resp.json()["code"] == "storage_error"
    assets = saved_assets(engine)
    assert len(assets) == 3 and all(a.file_url is None for a in assets)


def test_all_timeouts_is_504(client, mocks):
    mocks.fail.update({p: higgsfield.HiggsfieldTimeout("slow") for p in ("track", "beach", "city")})

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 504
    assert resp.json()["code"] == "image_timeout"


def test_unexpected_error_is_500_after_saving_paid_assets(client, engine, mocks):
    mocks.fail["beach"] = RuntimeError("bug")

    resp = client.post("/creative/generate", json=PAYLOAD)

    assert resp.status_code == 500
    assert "bug" not in resp.text
    assert sorted(a.format for a in saved_assets(engine)) == ["flat_lay", "ugc"]


@pytest.mark.parametrize("change", [{"goal": "retention"}, {"platform": "youtube"}, {"brief": ""}])
def test_invalid_request_is_400(client, mocks, change):
    resp = client.post("/creative/generate", json={**PAYLOAD, **change})

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_input"
    assert mocks.planner_calls == []


# --- angles ---

ANGLES = ["moms", "braids and locs care", "styling", "gym bag", "gifting"]


def angle_plan(angles, formats=None):
    formats = formats or ["lifestyle"] * len(angles)
    items = [make_item(f"a{i}", fmt) for i, fmt in enumerate(formats)]
    for item, angle in zip(items, angles):
        item["angle"] = angle
    return creative.CreativePlan.model_construct(items=[creative.CreativeItem(**i) for i in items])


def test_one_creative_per_angle_and_angle_saved(client, engine, mocks):
    mocks.plan = angle_plan(ANGLES[:2])

    resp = client.post("/creative/generate", json={**PAYLOAD, "angles": ANGLES[:2]})

    assert resp.status_code == 200, resp.text
    assert [a["angle"] for a in resp.json()["assets"]] == ANGLES[:2]
    assert [a.angle for a in saved_assets(engine)] == ANGLES[:2]
    assert len(mocks.generated) == 2
    assert "1. moms" in mocks.planner_calls[0]["messages"][0]["content"]


def test_single_angle_request_makes_one_creative_even_if_the_model_omits_the_angle(client, engine, mocks):
    mocks.plan = angle_plan([None], ["model_with_product"])  # the live failure: `angle` left out

    resp = client.post("/creative/generate", json={**PAYLOAD, "angles": ["peptide care"]})

    assert resp.status_code == 200, resp.text
    assert [(a["format"], a["angle"]) for a in resp.json()["assets"]] == [("model_with_product", "peptide care")]
    assert [a.angle for a in saved_assets(engine)] == ["peptide care"]
    assert len(mocks.planner_calls) == 1 and len(mocks.generated) == 1


def test_without_angles_assets_have_no_angle(client, engine, mocks):
    client.post("/creative/generate", json=PAYLOAD)

    assert [a.angle for a in saved_assets(engine)] == [None, None, None]


@pytest.mark.parametrize("angles, blocked", [(None, False), (ANGLES, True)])
def test_spend_check_scales_with_the_number_of_creatives(client, engine, mocks, angles, blocked):
    with Session(engine) as session:
        session.add(make_asset(cost_usd=Decimal("4.965")))
        session.commit()
    mocks.plan = angle_plan(angles) if angles else None

    # 3 creatives: 4.965 + 3 x 0.006 + 0.01 = 4.993 (ok); 5 angles: 4.965 + 0.03 + 0.01 = 5.005 (over)
    resp = client.post("/creative/generate", json={**PAYLOAD, "angles": angles})

    assert (resp.status_code == 402) == blocked



def test_source_text_without_rights_is_400(client, mocks):
    resp = client.post("/creative/generate", json={**PAYLOAD, "source_text": "A line from my book."})

    assert resp.status_code == 400
    assert "has_text_rights" in resp.text
    assert mocks.planner_calls == []


def test_rights_confirmation_saved_on_assets(client, engine, mocks):
    body = {**PAYLOAD, "source_text": "A line from my book.", "has_text_rights": True}

    assert client.post("/creative/generate", json=body).status_code == 200
    assert [a.text_rights_confirmed for a in saved_assets(engine)] == [True] * 3


def test_no_source_text_leaves_rights_unset(client, engine, mocks):
    client.post("/creative/generate", json=PAYLOAD)

    assert [a.text_rights_confirmed for a in saved_assets(engine)] == [None] * 3


def test_text_settings_saved_on_assets(client, engine, mocks):
    plan = make_plan()
    for item in plan.items:
        item.prompt += ", clean negative space in the top third"
    mocks.plan = plan

    body = {**PAYLOAD, "text_choice": "own", "on_image_text": "Trails are calling"}
    resp = client.post("/creative/generate", json=body)

    assert resp.status_code == 200, resp.text
    assets = saved_assets(engine)
    assert [(a.text_choice, a.text_mode, a.on_image_text) for a in assets] == [("own", "overlay", "Trails are calling")] * 3


def test_no_text_saves_choice_none_and_no_mode(client, engine, mocks):
    client.post("/creative/generate", json=PAYLOAD)

    assert [(a.text_choice, a.text_mode, a.on_image_text) for a in saved_assets(engine)] == [("none", None, None)] * 3


def test_bad_on_image_text_is_400_before_any_spend(client, mocks):
    body = {**PAYLOAD, "text_choice": "own", "on_image_text": "one two three four five six", "text_mode": "rendered"}

    resp = client.post("/creative/generate", json=body)

    assert resp.status_code == 400
    assert mocks.planner_calls == []


# --- creative tests ---

COMPOSITION = "on a sunlit trail at golden hour, mid-shot"


def creative_test_plan(angles):
    items = []
    for i, angle in enumerate(angles):
        item = make_item(f"person{i}, {COMPOSITION}", "ugc")
        item.update(angle=angle, core_message=f"{angle} message")
        items.append(creative.CreativeItem(**item))
    return creative.CreativePlan.model_construct(
        items=items, format_rationale="UGC feels native for every angle.", shared_composition=COMPOSITION,
    )


def test_creative_test_assets_share_a_test_group_id(client, engine, mocks):
    angles = ["moms", "braids and locs care", "styling"]
    mocks.plan = creative_test_plan(angles)

    resp = client.post("/creative/generate", json={**PAYLOAD, "creative_test": True, "angles": angles})

    assert resp.status_code == 200, resp.text
    body = resp.json()
    group = body["test_group_id"]
    assert group and len(group) == 32
    assert body["format_rationale"] == "UGC feels native for every angle."
    assets = saved_assets(engine)
    assert [a.test_group_id for a in assets] == [group] * 3
    assert body["test_mode"] == "fixed_format"
    assert [a.test_mode for a in assets] == ["fixed_format"] * 3
    assert [a.angle for a in assets] == angles
    assert [a.core_message for a in assets] == [f"{a} message" for a in angles]
    assert {a.format for a in assets} == {"ugc"}


def test_separate_tests_get_separate_group_ids(client, mocks):
    mocks.plan = creative_test_plan(["moms", "styling"])
    body = {**PAYLOAD, "creative_test": True, "angles": ["moms", "styling"]}

    first = client.post("/creative/generate", json=body).json()["test_group_id"]
    second = client.post("/creative/generate", json=body).json()["test_group_id"]

    assert first != second


def test_normal_run_has_no_test_group(client, engine, mocks):
    body = client.post("/creative/generate", json=PAYLOAD).json()

    assert body["test_group_id"] is None and body["format_rationale"] is None and body["test_mode"] is None
    assert [(a.test_group_id, a.core_message, a.test_mode) for a in saved_assets(engine)] == [(None, None, None)] * 3


def test_creative_test_without_angles_is_400_before_any_spend(client, mocks):
    resp = client.post("/creative/generate", json={**PAYLOAD, "creative_test": True})

    assert resp.status_code == 400
    assert "2-5 angles" in resp.text
    assert mocks.planner_calls == []



def test_varied_format_test_is_marked_varied(client, engine, mocks):
    plan = creative_test_plan(["moms", "styling"])
    lifestyle = make_item("person9", "lifestyle")
    lifestyle.update(angle="styling", core_message="styling message")
    plan.items[1] = creative.CreativeItem(**lifestyle)
    mocks.plan = plan

    body = {**PAYLOAD, "creative_test": True, "angles": ["moms", "styling"], "vary_format": True}
    resp = client.post("/creative/generate", json=body)

    assert resp.status_code == 200, resp.text
    assert resp.json()["test_mode"] == "varied_format"
    assets = saved_assets(engine)
    assert [(a.test_mode, a.angle, a.format) for a in assets] == [
        ("varied_format", "moms", "ugc"), ("varied_format", "styling", "lifestyle"),
    ]


def test_older_photo_is_read_once_and_its_reference_saved(client, engine, photo_mocks):
    photo_mocks.saved_reference = None  # uploaded before references existed

    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert photo_mocks.photo_reads == [b"photo-bytes"]
    assert photo_mocks.saved_references == [(PHOTO, REFERENCE)]  # saved as read; corrected per run
    assert {check[2] for check in photo_mocks.label_checks} == {CORRECTED}
    # the image + $0.004 check + ($0.00105 planning + $0.003 photo read) / 3
    for asset in saved_assets(engine):
        expected = higgsfield.product_image_cost(asset.prompt) + Decimal("0.004") + Decimal("0.00135")
        assert asset.cost_usd == expected.quantize(Decimal("0.0001"), ROUND_HALF_UP)


def test_saved_reference_means_no_new_read_of_the_photo(client, photo_mocks):
    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert photo_mocks.photo_reads == [] and photo_mocks.saved_references == []


def test_unreadable_photo_falls_back_to_the_brand_only_check(client, photo_mocks):
    photo_mocks.saved_reference, photo_mocks.photo_label = None, None

    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 200
    assert photo_mocks.saved_references == []
    assert {check[2] for check in photo_mocks.label_checks} == {None}


def test_every_image_prompt_gets_realism_and_one_product_rules(client, photo_mocks):
    client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    prompts = {key_of(prompt): prompt for prompt, _ in photo_mocks.product_calls}
    # make_plan: ugc "track" (people), lifestyle "beach" and flat_lay "city" (no people words)
    assert creative.REALISM_PROMPT in prompts["track"] and creative.ONE_PRODUCT_HELD in prompts["track"]
    for key in ("beach", "city"):
        assert creative.REALISM_PROMPT not in prompts[key]
        assert creative.ONE_PRODUCT in prompts[key] and creative.HOLD_LOWER_HALF not in prompts[key]
