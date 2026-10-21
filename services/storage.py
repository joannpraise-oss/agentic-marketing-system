import logging
import os
import uuid

import httpx

logger = logging.getLogger(__name__)

BUCKET = "assets"
TIMEOUT_SECONDS = 30


class StorageError(Exception):
    """Upload failed. The message never contains credentials or response bodies."""


def _config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise StorageError("Supabase storage is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
    return url, key


def upload_asset(data: bytes, content_type: str, extension: str = "") -> str:
    """Upload bytes to the public `assets` bucket and return the public URL."""
    base_url, key = _config()
    path = f"{uuid.uuid4().hex}{'.' + extension.lstrip('.') if extension else ''}"
    try:
        response = httpx.post(
            f"{base_url}/storage/v1/object/{BUCKET}/{path}",
            content=data,
            headers={"Authorization": f"Bearer {key}", "Content-Type": content_type},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logger.error("Storage upload rejected with status %s", exc.response.status_code)
        raise StorageError(f"Storage upload failed with status {exc.response.status_code}.") from None
    except httpx.HTTPError as exc:
        logger.error("Storage upload failed: %s", type(exc).__name__)
        raise StorageError("Storage upload failed.") from None
    return f"{base_url}/storage/v1/object/public/{BUCKET}/{path}"
