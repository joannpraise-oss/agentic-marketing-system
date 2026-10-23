import logging
import os
import time
import uuid
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

BUCKET = "assets"
TIMEOUT_SECONDS = 30
LEGACY_JWT_PREFIX = "eyJ"  # legacy service_role keys are JWTs; new keys look like sb_secret_...
MAX_LOGGED_ERROR_CHARS = 300
# Wait before each retry; retries happen only on network errors and 5xx, never on 4xx.
RETRY_DELAYS_SECONDS = (0.5, 1.5)


class StorageError(Exception):
    """Upload failed. The message never contains credentials or response bodies."""


def check_supabase_url() -> None:
    """Fail fast at startup on a REST endpoint URL instead of the project URL."""
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    if url.endswith("/rest/v1"):
        raise RuntimeError(
            "SUPABASE_URL must be the project URL (https://<ref>.supabase.co), not the /rest/v1 endpoint."
        )


def _config() -> tuple[str, str]:
    url = os.getenv("SUPABASE_URL", "").rstrip("/")
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")
    if not url or not key:
        raise StorageError("Supabase storage is not configured (SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY).")
    if url.endswith("/rest/v1"):
        raise StorageError("SUPABASE_URL must not end in /rest/v1.")
    return url, key


def _auth_headers(key: str) -> dict[str, str]:
    # New sb_secret_ keys go only in `apikey`; only legacy JWT keys may also be sent as a Bearer token.
    headers = {"apikey": key}
    if key.startswith(LEGACY_JWT_PREFIX):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _error_message(response: httpx.Response, key: str) -> str:
    """Supabase's own explanation: only the `error` / `message` fields, never the key."""
    try:
        body = response.json()
    except ValueError:
        return ""
    if not isinstance(body, dict):
        return ""
    parts = [str(body[field]) for field in ("error", "message") if body.get(field)]
    return ": ".join(parts).replace(key, "[redacted]")[:MAX_LOGGED_ERROR_CHARS]


def upload_asset(
    data: bytes, content_type: str, extension: str = "", *, sleep: Callable[[float], None] | None = None
) -> str:
    """Upload bytes to the public `assets` bucket and return the public URL."""
    sleep = sleep or time.sleep
    base_url, key = _config()
    path = f"{uuid.uuid4().hex}{'.' + extension.lstrip('.') if extension else ''}"
    # x-upsert: if an attempt landed but its response was lost, the retry overwrites the
    # same (random, unique) path instead of failing as a duplicate.
    headers = {**_auth_headers(key), "Content-Type": content_type, "x-upsert": "true"}
    attempts = len(RETRY_DELAYS_SECONDS) + 1
    for attempt in range(1, attempts + 1):
        try:
            response = httpx.post(
                f"{base_url}/storage/v1/object/{BUCKET}/{path}",
                content=data, headers=headers, timeout=TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            return f"{base_url}/storage/v1/object/public/{BUCKET}/{path}"
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            logger.error(
                "Storage upload rejected with status %s (attempt %s/%s): %s",
                status, attempt, attempts, _error_message(exc.response, key) or "no error message",
            )
            error = StorageError(f"Storage upload failed with status {status}.")
            if status < 500:
                raise error from None
        except httpx.HTTPError as exc:
            logger.error("Storage upload failed (attempt %s/%s): %s", attempt, attempts, type(exc).__name__)
            error = StorageError(f"Storage upload failed: network error ({type(exc).__name__}).")
        if attempt < attempts:
            sleep(RETRY_DELAYS_SECONDS[attempt - 1])
    raise error
