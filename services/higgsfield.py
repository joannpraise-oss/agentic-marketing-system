import logging
import os
import random
import time
import uuid
from dataclasses import dataclass
from decimal import Decimal
from typing import Callable

import httpx

logger = logging.getLogger(__name__)

BASE_URL = "https://api.higgsfield.ai"
PROVIDER = "higgsfield"
MODEL = "higgsfield-ai/soul/v2/standard"
IMAGE_PARAMS = {"aspect_ratio": "1:1", "resolution": "1080p", "num_images": 1}

HTTP_TIMEOUT_SECONDS = 30
GENERATION_TIMEOUT_SECONDS = 120
# Polling schedule from the Higgsfield docs: start at 2s, x1.5 per poll, cap at 10s, plus jitter.
POLL_START_SECONDS = 2.0
POLL_MAX_SECONDS = 10.0
POLL_JITTER_SECONDS = 0.5
TERMINAL_STATUSES = {"completed", "failed", "nsfw", "canceled"}


class HiggsfieldError(Exception):
    """The provider failed or returned something unusable. Messages never contain credentials."""


class HiggsfieldTimeout(HiggsfieldError):
    """The job did not reach a terminal status within the generation timeout."""


class HiggsfieldAuthError(HiggsfieldError):
    """Missing or rejected credentials: our problem, not the caller's."""


@dataclass
class GeneratedImage:
    data: bytes
    content_type: str
    request_id: str


def _auth_headers() -> dict[str, str]:
    # HIGGSFIELD_API_KEY holds "<key_id>:<key_secret>", as the Authorization header expects.
    key = os.getenv("HIGGSFIELD_API_KEY", "")
    if ":" not in key:
        raise HiggsfieldAuthError("Higgsfield is not configured (HIGGSFIELD_API_KEY must be '<key_id>:<key_secret>').")
    return {"Authorization": f"Key {key}"}


def _call(method: str, path: str, **kwargs) -> dict:
    headers = {**_auth_headers(), **kwargs.pop("headers", {})}
    try:
        response = httpx.request(
            method, f"{BASE_URL}{path}", headers=headers, timeout=HTTP_TIMEOUT_SECONDS, **kwargs
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status in (401, 403):
            raise HiggsfieldAuthError(f"Higgsfield rejected our credentials ({status}).") from None
        logger.error("Higgsfield %s %s returned status %s", method, path, status)
        raise HiggsfieldError(f"Higgsfield returned status {status}.") from None
    except httpx.TimeoutException:
        raise HiggsfieldError("Higgsfield did not respond in time.") from None
    except httpx.HTTPError as exc:
        logger.error("Higgsfield %s %s failed: %s", method, path, type(exc).__name__)
        raise HiggsfieldError("Could not reach Higgsfield.") from None
    except ValueError:
        raise HiggsfieldError("Higgsfield returned invalid JSON.") from None


def estimate_cost(prompt: str = "cost estimate") -> Decimal:
    """USD cost of one image with IMAGE_PARAMS. Higgsfield only charges completed jobs."""
    body = _call("POST", f"/estimate/{MODEL}", json={"prompt": prompt, **IMAGE_PARAMS})
    try:
        return Decimal(str(body["usd"]))
    except (KeyError, ArithmeticError):
        raise HiggsfieldError("Higgsfield estimate had no usable 'usd' value.") from None


def _cancel_quietly(request_id: str) -> None:
    # Only queued jobs can be canceled (and refunded); anything else is best-effort.
    try:
        _call("POST", f"/requests/{request_id}/cancel")
    except HiggsfieldError:
        pass


def _download(url: str) -> tuple[bytes, str]:
    # Output URLs are public CDN links: never send our credentials there.
    try:
        response = httpx.get(url, timeout=HTTP_TIMEOUT_SECONDS, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        logger.error("Higgsfield image download failed: %s", type(exc).__name__)
        raise HiggsfieldError("Could not download the generated image.") from None
    content_type = response.headers.get("content-type", "image/jpeg").split(";")[0]
    return response.content, content_type


def generate_image(
    prompt: str,
    *,
    timeout: float = GENERATION_TIMEOUT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> GeneratedImage:
    """Submit a Soul v2 job, poll until it finishes, and return the image bytes."""
    job = _call(
        "POST", f"/{MODEL}",
        json={"prompt": prompt, **IMAGE_PARAMS},
        headers={"Idempotency-Key": uuid.uuid4().hex},
    )
    request_id = job.get("request_id")
    if not request_id:
        raise HiggsfieldError("Higgsfield did not return a request id.")

    deadline = clock() + timeout
    delay = POLL_START_SECONDS
    status = job
    while status.get("status") not in TERMINAL_STATUSES:
        remaining = deadline - clock()
        if remaining <= 0:
            _cancel_quietly(request_id)
            logger.warning("Higgsfield job %s timed out after %ss", request_id, timeout)
            raise HiggsfieldTimeout(f"Image generation did not finish within {timeout:.0f}s.")
        sleep(min(delay + random.uniform(0, POLL_JITTER_SECONDS), remaining))
        delay = min(delay * 1.5, POLL_MAX_SECONDS)
        status = _call("GET", f"/requests/{request_id}/status")

    if status["status"] != "completed":
        logger.warning("Higgsfield job %s ended as %s: %s", request_id, status["status"], status.get("error"))
        raise HiggsfieldError(f"Image generation ended as '{status['status']}'.")

    images = status.get("images") or []
    if not images or not images[0].get("url"):
        raise HiggsfieldError("Higgsfield completed without an image URL.")

    data, content_type = _download(images[0]["url"])
    return GeneratedImage(data=data, content_type=content_type, request_id=request_id)
