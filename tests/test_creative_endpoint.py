import time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from db.models import Asset
from services import creative, higgsfield, storage
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
        key = prompt.split()[-1]  # make_item prompts end with a short key: "track", "beach", "city"
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


@pytest.fixture
def photo_mocks(monkeypatch, mocks):
    """Adds the signed URL and Marketing Studio to `mocks`; records (prompt, reference_url) calls."""
    mocks.product_calls, mocks.signed = [], []

    def create_signed_url(bucket, path):
        mocks.signed.append((bucket, path))
        return SIGNED

    def generate_product_image(prompt, reference_url):
        mocks.product_calls.append((prompt, reference_url))
        key = prompt.removesuffix(LABEL).split()[-1]
        return higgsfield.GeneratedImage(data=f"img:{key}".encode(), content_type="image/png", request_id=f"r-{key}")

    monkeypatch.setattr(storage, "create_signed_url", create_signed_url)
    monkeypatch.setattr(higgsfield, "generate_product_image", generate_product_image)
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
        # $0.035 image + $0.00105 / 3 planning = $0.03535, rounded to 4 places
        assert asset.cost_usd == Decimal("0.0354")


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
        session.add(make_asset(cost_usd=Decimal("4.92")))  # + $0.115 with a photo > 5.00; + $0.028 without
        session.commit()

    resp = client.post("/creative/generate", json={**PAYLOAD, "product_photo_path": PHOTO})

    assert resp.status_code == 402
    assert photo_mocks.signed == [] and photo_mocks.planner_calls == [] and photo_mocks.product_calls == []
    assert client.post("/creative/generate", json=PAYLOAD).status_code == 200


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


@pytest.mark.parametrize("error, code", [
    (higgsfield.HiggsfieldTimeout("Image generation did not finish within 120s."), "image_timeout"),
    (higgsfield.HiggsfieldError("Image generation ended as 'nsfw'."), "image_provider_error"),
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
