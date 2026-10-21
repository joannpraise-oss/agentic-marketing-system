import logging

import httpx
import pytest

from services import storage

KEY = "service-role-secret-123"


@pytest.fixture(autouse=True)
def supabase_env(monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://proj.supabase.co/")
    monkeypatch.setenv("SUPABASE_SERVICE_ROLE_KEY", KEY)


def mock_post(monkeypatch, status=200, raises=None):
    calls = []

    def fake_post(url, **kwargs):
        calls.append((url, kwargs))
        if raises:
            raise raises
        return httpx.Response(status, request=httpx.Request("POST", url))

    monkeypatch.setattr(storage.httpx, "post", fake_post)
    return calls


def test_upload_returns_public_url_and_sends_auth(monkeypatch):
    calls = mock_post(monkeypatch)

    url = storage.upload_asset(b"png-bytes", "image/png", "png")

    post_url, kwargs = calls[0]
    assert post_url.startswith("https://proj.supabase.co/storage/v1/object/assets/")
    assert post_url.endswith(".png")
    assert kwargs["content"] == b"png-bytes"
    assert kwargs["headers"]["Authorization"] == f"Bearer {KEY}"
    assert url == post_url.replace("/object/assets/", "/object/public/assets/")


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


def test_network_error_raises_storage_error(monkeypatch):
    mock_post(monkeypatch, raises=httpx.ConnectError("boom"))
    with pytest.raises(storage.StorageError):
        storage.upload_asset(b"x", "image/png")
