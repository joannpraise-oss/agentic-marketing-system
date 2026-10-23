import time
from decimal import Decimal
from types import SimpleNamespace

import pytest
from sqlmodel import Session, select

from db.models import Asset
from services import creative, higgsfield, storage
from tests.test_asset_model import make_asset
from tests.test_creative import make_plan

PAYLOAD = {
    "brief": "Spring sale on trail runners",
    "brand": {
        "brand_name": "Acme", "voice": "playful", "colors": ["red", "white"],
        "product": "trail running shoes", "audience": "weekend runners",
    },
    "goal": "sales",
    "platform": "instagram",
}


@pytest.fixture
def mocks(monkeypatch):
    """OpenAI, Higgsfield and storage all faked. Set `fail` / `upload_fail` to a prompt -> exception."""
    state = SimpleNamespace(
        planner_calls=[], generated=[], uploads=[], fail={}, upload_fail={},
        active_uploads=0, max_parallel_uploads=0,
    )

    def create_with_completion(**kwargs):
        state.planner_calls.append(kwargs)
        # 3000 in / 1000 out tokens = 0.00045 + 0.0006 = $0.00105 planning cost
        usage = SimpleNamespace(prompt_tokens=3000, completion_tokens=1000, total_tokens=4000)
        return make_plan(), SimpleNamespace(usage=usage)

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
