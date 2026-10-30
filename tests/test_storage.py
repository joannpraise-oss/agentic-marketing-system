import logging
import re

import httpx
import pytest

from services import storage

KEY = "sb_secret_abc123"
LEGACY_KEY = "eyJhbGciOiJIUzI1NiJ9.legacy-service-role.sig"


@pytest.fixture(autouse=True)
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", KEY)


def mock_post(monkeypatch, status=200, raises=None, outcomes=None, **response_kwargs):
    """Fake httpx.post. `outcomes` scripts successive attempts: an int status or an exception."""
    calls = []
    script = list(outcomes) if outcomes is not None else None

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        outcome = script.pop(0) if script else (raises or status)
        if isinstance(outcome, Exception):
            raise outcome
        return httpx.Response(outcome, request=httpx.Request("POST", url), **response_kwargs)

    monkeypatch.setattr(storage.httpx, "post", fake_post)
    return calls


@pytest.fixture(autouse=True)
def no_real_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(storage.time, "sleep", sleeps.append)
    return sleeps


def test_upload_returns_public_url_and_sends_auth(monkeypatch):
    calls = mock_post(monkeypatch)

    url = storage.upload_asset(b"png-bytes", "image/png", "png")

    post_url, kwargs = calls[0]
    assert post_url.startswith("https://proj.supabase.co/storage/v1/object/assets/")
    assert post_url.endswith(".png")
    assert kwargs["content"] == b"png-bytes"
    assert url == post_url.replace("/object/assets/", "/object/public/assets/")


def test_new_secret_key_goes_only_in_apikey_header(monkeypatch):
    calls = mock_post(monkeypatch)

    storage.upload_asset(b"x", "image/png")

    headers = calls[0][1]["headers"]
    assert headers["apikey"] == KEY
    assert "Authorization" not in headers


def test_legacy_jwt_key_also_sent_as_bearer(monkeypatch):
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", LEGACY_KEY)
    calls = mock_post(monkeypatch)

    storage.upload_asset(b"x", "image/png")

    headers = calls[0][1]["headers"]
    assert headers["apikey"] == LEGACY_KEY
    assert headers["Authorization"] == f"Bearer {LEGACY_KEY}"


def test_missing_config_raises(monkeypatch):
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY")
    with pytest.raises(storage.StorageError, match="not configured"):
        storage.upload_asset(b"x", "image/png")


def test_http_error_raises_without_leaking_key(monkeypatch, caplog):
    mock_post(monkeypatch, status=403)

    with caplog.at_level(logging.DEBUG), pytest.raises(storage.StorageError) as info:
        storage.upload_asset(b"x", "image/png")

    assert "403" in str(info.value)
    assert KEY not in str(info.value) and KEY not in caplog.text


def test_http_error_logs_supabase_error_and_message_only(monkeypatch, caplog):
    body = {
        "statusCode": "400", "error": "InvalidRequest", "message": f"bad header {KEY}",
        "secret_debug": "do-not-log-me",
    }
    mock_post(monkeypatch, status=400, json=body)

    with caplog.at_level(logging.DEBUG), pytest.raises(storage.StorageError) as info:
        storage.upload_asset(b"x", "image/png")

    assert "InvalidRequest: bad header [redacted]" in caplog.text
    assert "do-not-log-me" not in caplog.text
    assert KEY not in caplog.text and KEY not in str(info.value)


def test_http_error_with_non_json_body_still_logs_status(monkeypatch, caplog):
    mock_post(monkeypatch, status=502, text="<html>bad gateway</html>")

    with caplog.at_level(logging.DEBUG), pytest.raises(storage.StorageError):
        storage.upload_asset(b"x", "image/png")

    assert re.search(r"status 502 after [\d.]+s \(attempt 1/3\): no error message", caplog.text)
    assert "<html>" not in caplog.text


@pytest.mark.parametrize("url", ["https://proj.supabase.co/rest/v1", "https://proj.supabase.co/rest/v1/"])
def test_rest_v1_url_is_rejected(monkeypatch, url):
    monkeypatch.setenv("SUPABASE_URL", url)
    calls = mock_post(monkeypatch)

    with pytest.raises(RuntimeError, match="rest/v1"):
        storage.check_supabase_url()
    with pytest.raises(storage.StorageError, match="rest/v1"):
        storage.upload_asset(b"x", "image/png")
    assert calls == []


@pytest.mark.parametrize("url", ["https://proj.supabase.co", "https://proj.supabase.co/", ""])
def test_project_url_or_unset_passes_startup_check(monkeypatch, url):
    monkeypatch.setenv("SUPABASE_URL", url)
    storage.check_supabase_url()


def test_app_startup_fails_on_rest_v1_url(monkeypatch):
    from fastapi.testclient import TestClient

    from main import app

    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co/rest/v1")
    with pytest.raises(RuntimeError, match="rest/v1"):
        with TestClient(app):
            pass


def test_network_error_raises_storage_error(monkeypatch):
    mock_post(monkeypatch, raises=httpx.ConnectError("boom"))
    with pytest.raises(storage.StorageError):
        storage.upload_asset(b"x", "image/png")


# --- retries ---

def test_network_error_then_success_retries_and_returns_url(monkeypatch, no_real_sleep):
    calls = mock_post(monkeypatch, outcomes=[httpx.ConnectError("reset"), 200])

    url = storage.upload_asset(b"x", "image/png", "png")

    assert len(calls) == 2
    assert calls[0][0] == calls[1][0]  # same path, so a lost-response retry overwrites, not duplicates
    assert calls[0][1]["headers"]["x-upsert"] == "true"
    assert url.endswith(".png")
    assert no_real_sleep == [storage.RETRY_DELAYS_SECONDS[0]]


def test_5xx_retried_up_to_two_times_then_fails(monkeypatch, no_real_sleep):
    calls = mock_post(monkeypatch, outcomes=[503, 502, 500])

    with pytest.raises(storage.StorageError, match="status 500"):
        storage.upload_asset(b"x", "image/png")

    assert len(calls) == 3
    assert no_real_sleep == list(storage.RETRY_DELAYS_SECONDS)


def test_5xx_then_success(monkeypatch):
    calls = mock_post(monkeypatch, outcomes=[500, 500, 200])

    assert storage.upload_asset(b"x", "image/png")
    assert len(calls) == 3


def test_repeated_network_errors_give_up_after_three_attempts(monkeypatch):
    calls = mock_post(monkeypatch, raises=httpx.ReadTimeout("slow"))

    with pytest.raises(storage.StorageError, match="network error"):
        storage.upload_asset(b"x", "image/png")

    assert len(calls) == 3


@pytest.mark.parametrize("status", [400, 401, 403, 404, 409])
def test_4xx_is_not_retried(monkeypatch, no_real_sleep, status):
    calls = mock_post(monkeypatch, outcomes=[status, 200])

    with pytest.raises(storage.StorageError, match=f"status {status}"):
        storage.upload_asset(b"x", "image/png")

    assert len(calls) == 1
    assert no_real_sleep == []


# --- upload diagnostics ---

def test_success_logs_size_in_kb_and_duration(monkeypatch, caplog):
    mock_post(monkeypatch)

    with caplog.at_level(logging.INFO):
        storage.upload_asset(b"x" * 2048, "image/png", "png")

    assert re.search(r"Uploaded assets/\w+\.png \(2\.0 KB\) in [\d.]+s \(attempt 1/3\)", caplog.text)


def test_each_failed_attempt_logs_size_and_duration(monkeypatch, caplog):
    mock_post(monkeypatch, outcomes=[httpx.ReadError("reset"), httpx.ReadError("reset"), 200])

    with caplog.at_level(logging.INFO):
        storage.upload_asset(b"x" * 1536, "image/png")

    failures = re.findall(r"\(1\.5 KB\) failed after [\d.]+s \(attempt (\d)/3\): ReadError", caplog.text)
    assert failures == ["1", "2"]
    assert "attempt 3/3" in caplog.text


def test_files_over_6mb_log_a_warning(monkeypatch, caplog):
    mock_post(monkeypatch)
    monkeypatch.setattr(storage, "STANDARD_UPLOAD_MAX_BYTES", 1024)

    with caplog.at_level(logging.INFO):
        storage.upload_asset(b"x" * 2048, "image/png")

    assert "over Supabase's 6 MB standard-upload guidance" in caplog.text


def test_upload_timeout_is_60s(monkeypatch):
    calls = mock_post(monkeypatch)

    storage.upload_asset(b"x", "image/png")

    assert calls[0][1]["timeout"] == 60


# --- private product photos and signed URLs ---

def test_product_photo_goes_to_the_private_bucket(monkeypatch):
    calls = mock_post(monkeypatch)

    path = storage.upload_product_photo(b"jpeg-bytes")

    url, kwargs = calls[0]
    assert url == f"https://proj.supabase.co/storage/v1/object/product-photos/{path}"
    assert path.endswith(".jpg")
    assert kwargs["headers"]["Content-Type"] == "image/jpeg"
    assert kwargs["headers"]["apikey"] == KEY


def test_signed_url_lasts_ten_minutes_and_is_absolute(monkeypatch, caplog):
    calls = []
    token_path = "/object/sign/product-photos/abc.jpg?token=SECRET-TOKEN"

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        return httpx.Response(200, json={"signedURL": token_path}, request=httpx.Request("POST", url))

    monkeypatch.setattr(storage.httpx, "post", fake_post)

    with caplog.at_level(logging.DEBUG):
        url = storage.create_signed_url("product-photos", "abc.jpg")

    assert calls[0][0] == "https://proj.supabase.co/storage/v1/object/sign/product-photos/abc.jpg"
    assert calls[0][1]["json"] == {"expiresIn": 600}
    assert url == f"https://proj.supabase.co/storage/v1{token_path}"
    assert "SECRET-TOKEN" not in caplog.text


@pytest.mark.parametrize("status, body, not_found", [
    (400, {"statusCode": "404", "error": "not_found", "message": "Object not found"}, True),  # what Supabase sends
    (404, {}, True),
    (400, {"error": "invalid"}, False),
    (403, {"error": "forbidden"}, False),
])
def test_signing_a_missing_object_is_reported_as_not_found(monkeypatch, status, body, not_found):
    monkeypatch.setattr(
        storage.httpx, "post",
        lambda url, **kwargs: httpx.Response(status, json=body, request=httpx.Request("POST", url)),
    )

    with pytest.raises(storage.StorageError) as exc:
        storage.create_signed_url("product-photos", "abc.jpg")

    assert isinstance(exc.value, storage.StorageObjectNotFound) is not_found


@pytest.mark.parametrize("outcome, message", [
    (404, "status 404"),
    (httpx.ConnectError("down"), "network error"),
    ("no-field", "no signed URL"),
])
def test_signed_url_failures_raise_storage_error(monkeypatch, outcome, message):
    def fake_post(url, **kwargs):
        if isinstance(outcome, Exception):
            raise outcome
        if outcome == "no-field":
            return httpx.Response(200, json={}, request=httpx.Request("POST", url))
        return httpx.Response(outcome, json={"error": "not_found"}, request=httpx.Request("POST", url))

    monkeypatch.setattr(storage.httpx, "post", fake_post)

    with pytest.raises(storage.StorageError, match=message):
        storage.create_signed_url("product-photos", "abc.jpg")


def mock_request(monkeypatch, status=200, content=b""):
    calls = []

    def fake_request(method, url, **kwargs):
        calls.append((method, url, kwargs))
        return httpx.Response(status, content=content, request=httpx.Request(method, url))

    monkeypatch.setattr(storage.httpx, "request", fake_request)
    return calls


def test_download_product_photo_uses_authenticated_endpoint(monkeypatch):
    calls = mock_request(monkeypatch, content=b"jpeg-bytes")

    assert storage.download_product_photo("abc.jpg") == b"jpeg-bytes"
    method, url, kwargs = calls[0]
    assert (method, url) == ("GET", "https://proj.supabase.co/storage/v1/object/authenticated/product-photos/abc.jpg")
    assert kwargs["headers"]["apikey"] == KEY


def test_delete_product_photo(monkeypatch):
    calls = mock_request(monkeypatch)

    storage.delete_product_photo("abc.jpg")

    assert calls[0][:2] == ("DELETE", "https://proj.supabase.co/storage/v1/object/product-photos/abc.jpg")


@pytest.mark.parametrize("call", [storage.download_product_photo, storage.delete_product_photo])
def test_private_request_failures_raise_storage_error(monkeypatch, call):
    mock_request(monkeypatch, status=404)

    with pytest.raises(storage.StorageError, match="status 404"):
        call("abc.jpg")
