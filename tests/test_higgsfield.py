import logging
from decimal import Decimal

import httpx
import pytest
from fastapi import APIRouter

from main import app
from services import higgsfield
from services.storage import StorageError

KEY = "key-id-123:key-secret-456"
IMAGE_URL = "https://cdn.higgsfield.ai/out/abc.png"


@pytest.fixture(autouse=True)
def higgsfield_env(monkeypatch):
    monkeypatch.setenv("HIGGSFIELD_API_KEY", KEY)


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeHiggsfield:
    """Scripted httpx.request / httpx.get: `statuses` are returned by successive status polls."""

    def __init__(self, statuses=("completed",), submit_status=200, image=b"png-bytes"):
        self.statuses = list(statuses)
        self.submit_status = submit_status
        self.image = image
        self.calls = []
        self.downloads = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        req = httpx.Request(method, url)
        if url.endswith(("/" + higgsfield.MODEL, "/" + higgsfield.PRODUCT_MODEL)):
            return httpx.Response(self.submit_status, json={"status": "queued", "request_id": "r1"}, request=req)
        if url.endswith("/cancel"):
            return httpx.Response(200, json={}, request=req)
        status = self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]
        body = {"status": status, "request_id": "r1"}
        if status == "completed":
            body["images"] = [{"url": IMAGE_URL}]
        if status == "failed":
            body["error"] = "Generation failed"
        return httpx.Response(200, json=body, request=req)

    def get(self, url, **kwargs):
        self.downloads.append((url, kwargs))
        return httpx.Response(200, content=self.image, headers={"content-type": "image/png"}, request=httpx.Request("GET", url))


@pytest.fixture
def fake(monkeypatch):
    fake = FakeHiggsfield()
    monkeypatch.setattr(higgsfield.httpx, "request", fake.request)
    monkeypatch.setattr(higgsfield.httpx, "get", fake.get)
    return fake


def run(clock, **kwargs):
    return higgsfield.generate_image("a red shoe", sleep=clock.sleep, clock=clock, **kwargs)


def test_generate_polls_until_completed_and_downloads(fake):
    fake.statuses = ["queued", "in_progress", "completed"]
    clock = FakeClock()

    image = run(clock)

    assert image.data == b"png-bytes"
    assert image.content_type == "image/png"
    assert image.request_id == "r1"
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ("POST", f"{higgsfield.BASE_URL}/{higgsfield.MODEL}")
    assert kwargs["json"]["prompt"] == "a red shoe"
    assert kwargs["headers"]["Authorization"] == f"Key {KEY}"
    assert kwargs["headers"]["Idempotency-Key"]
    assert [c[1] for c in fake.calls[1:]] == [f"{higgsfield.BASE_URL}/requests/r1/status"] * 3
    # The image CDN never sees our credentials.
    assert "headers" not in fake.downloads[0][1]


def test_submit_sends_supported_resolution(fake):
    # Soul v2 standard only accepts "720p" / "1080p"; "2K" is rejected by the live API.
    assert higgsfield.IMAGE_PARAMS["resolution"] == "1080p"

    run(FakeClock())

    assert fake.calls[0][2]["json"]["resolution"] == "1080p"


SIGNED = "https://proj.supabase.co/storage/v1/object/sign/product-photos/p.jpg?token=SECRET-TOKEN"


def test_product_image_uses_marketing_studio_direct_low_1k_with_the_reference(fake):
    clock = FakeClock()

    image = higgsfield.generate_product_image("a bottle", SIGNED, sleep=clock.sleep, clock=clock)

    assert image.data == b"png-bytes"
    method, url, kwargs = fake.calls[0]
    assert (method, url) == ("POST", f"{higgsfield.BASE_URL}/marketing-studio/image")
    # Exactly these fields: the model rejects unknown ones, such as Soul's num_images.
    assert kwargs["json"] == {
        "prompt": "a bottle", "image_urls": [SIGNED],
        "quality": "low", "resolution": "1k", "aspect_ratio": "1:1", "enhance_prompt": False,
    }
    assert kwargs["headers"]["Idempotency-Key"]


def test_product_image_failure_never_logs_the_signed_url(fake, caplog):
    fake.submit_status = 500
    clock = FakeClock()

    with caplog.at_level(logging.DEBUG), pytest.raises(higgsfield.HiggsfieldError) as exc:
        higgsfield.generate_product_image("a bottle", SIGNED, sleep=clock.sleep, clock=clock)

    assert "SECRET-TOKEN" not in caplog.text and "SECRET-TOKEN" not in str(exc.value)


# Higgsfield's own /estimate prices on 2026-11-03, by prompt length.
@pytest.mark.parametrize("chars, higgsfield_usd", [(1000, "0.035"), (1500, "0.037"), (1897, "0.039"),
                                                   (2398, "0.042"), (3795, "0.049")])
def test_product_image_cost_matches_or_slightly_exceeds_the_real_price(chars, higgsfield_usd):
    cost = higgsfield.product_image_cost("x" * chars)

    assert Decimal(higgsfield_usd) <= cost <= Decimal(higgsfield_usd) + Decimal("0.0015")


def test_product_image_reserve_covers_a_3000_character_prompt(fake):
    assert higgsfield.estimate_product_image_cost() == higgsfield.product_image_cost("x" * 3000) == Decimal("0.0456")
    assert fake.calls == []


def test_poll_delay_backs_off_and_caps(fake):
    fake.statuses = ["queued"] * 8 + ["completed"]
    clock = FakeClock()

    run(clock)

    sleeps = clock.sleeps  # each sleep is the backoff delay plus up to 0.5s jitter
    assert 2.0 <= sleeps[0] <= 2.5
    assert 3.0 <= sleeps[1] <= 3.5
    assert all(s <= higgsfield.POLL_MAX_SECONDS + higgsfield.POLL_JITTER_SECONDS for s in sleeps)
    assert max(sleeps) >= higgsfield.POLL_MAX_SECONDS


def test_timeout_cancels_and_raises(fake):
    fake.statuses = ["queued"]
    clock = FakeClock()

    with pytest.raises(higgsfield.HiggsfieldTimeout):
        run(clock, timeout=30)

    assert clock.now == pytest.approx(30)  # never sleeps past the deadline
    assert fake.calls[-1][1].endswith("/requests/r1/cancel")
    assert fake.downloads == []


@pytest.mark.parametrize("status", ["failed", "nsfw", "canceled"])
def test_non_completed_terminal_status_raises_provider_error(fake, status):
    fake.statuses = [status]

    with pytest.raises(higgsfield.HiggsfieldError) as info:
        run(FakeClock())

    assert not isinstance(info.value, higgsfield.HiggsfieldTimeout)
    assert isinstance(info.value, higgsfield.HiggsfieldContentFiltered) is (status == "nsfw")
    assert status in str(info.value)


def test_content_filter_rejection_logs_that_it_was_not_charged(fake, caplog):
    fake.statuses = ["nsfw"]

    with caplog.at_level(logging.WARNING), pytest.raises(higgsfield.HiggsfieldContentFiltered):
        run(FakeClock())

    assert "content filter" in caplog.text and "charged: no" in caplog.text


@pytest.mark.parametrize("http_status, error", [(401, higgsfield.HiggsfieldAuthError), (500, higgsfield.HiggsfieldError)])
def test_submit_http_errors(fake, http_status, error, caplog):
    fake.submit_status = http_status

    with caplog.at_level(logging.DEBUG), pytest.raises(error) as info:
        run(FakeClock())

    assert "key-secret-456" not in str(info.value) and "key-secret-456" not in caplog.text


def test_missing_key_is_auth_error(fake, monkeypatch):
    monkeypatch.delenv("HIGGSFIELD_API_KEY")
    with pytest.raises(higgsfield.HiggsfieldAuthError):
        run(FakeClock())
    assert fake.calls == []


def test_estimate_cost_is_live_checked_price_without_api_call(fake):
    # Pinned from the 2026-10-21 live /estimate check at 1:1, 1080p, 1 image.
    assert higgsfield.estimate_cost() == Decimal("0.006")
    assert fake.calls == []


# --- error mapping through core/errors.py ---

router = APIRouter()
ERRORS = {
    "timeout": higgsfield.HiggsfieldTimeout("slow"),
    "provider": higgsfield.HiggsfieldError("bad"),
    "auth": higgsfield.HiggsfieldAuthError("key-secret-456 rejected"),
    "storage": StorageError("upload failed"),
}


@router.get("/_test/image-error/{name}")
def raise_image_error(name: str):
    raise ERRORS[name]


app.include_router(router)


@pytest.mark.parametrize("name, status, code", [
    ("timeout", 504, "image_timeout"),
    ("provider", 502, "image_provider_error"),
    ("auth", 500, "internal_error"),
    ("storage", 502, "storage_error"),
])
def test_image_errors_map_to_standard_shape(client, name, status, code):
    resp = client.get(f"/_test/image-error/{name}")

    assert resp.status_code == status
    body = resp.json()
    assert body["code"] == code
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert "key-secret-456" not in resp.text
