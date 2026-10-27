"""One-off check: does a Higgsfield model keep a real product from a reference photo?

Prints the /estimate price first and asks before spending anything. Then it makes
ONE generation with a 10-minute signed URL to a photo in the private
product-photos bucket, and saves the result in ./reference_check/ to compare by eye.

Usage (from the repo root, with .env set up):

    venv/bin/python scripts/try_reference_model.py \
        --model-path <path from console.higgsfield.ai, e.g. vendor/model/variant> \
        --field image_urls \
        --photo-path <product_photo_path returned by POST /creative/product-photo> \
        --prompt "Studio hero shot of this exact product on warm linen, soft daylight" \
        --param aspect_ratio=1:1

--field is the reference-image field name from the model's console page.
Use --field-shape list (default) for an array of URLs, or string for one URL.
Credentials and the signed URL's token are never printed.
"""
import argparse
import json
import random
import sys
import time
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from services import higgsfield, storage  # noqa: E402  (needs .env loaded first)

OUTPUT_DIR = Path("reference_check")
POLL_TIMEOUT_SECONDS = 180


def build_body(prompt: str, field: str, field_shape: str, signed_url: str, params: list[str]) -> dict:
    body: dict = {"prompt": prompt}
    for param in params:
        key, _, value = param.partition("=")
        try:
            body[key] = json.loads(value)  # numbers, true/false, lists
        except json.JSONDecodeError:
            body[key] = value  # plain strings like 1:1 or 1080p
    body[field] = [signed_url] if field_shape == "list" else signed_url
    return body


def redacted(body: dict, field: str) -> dict:
    shown = dict(body)
    shown[field] = "<signed URL, 10 min>" if isinstance(body[field], str) else ["<signed URL, 10 min>"]
    return shown


def wait_for_result(request_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_SECONDS
    delay = higgsfield.POLL_START_SECONDS
    while True:
        status = higgsfield._call("GET", f"/requests/{request_id}/status")
        print(f"  status: {status.get('status')}")
        if status.get("status") in higgsfield.TERMINAL_STATUSES:
            return status
        if time.monotonic() > deadline:
            sys.exit(f"Gave up after {POLL_TIMEOUT_SECONDS}s; request {request_id} is still {status.get('status')}.")
        time.sleep(delay + random.uniform(0, higgsfield.POLL_JITTER_SECONDS))
        delay = min(delay * 1.5, higgsfield.POLL_MAX_SECONDS)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model-path", required=True, help="Model path on api.higgsfield.ai, without leading slash")
    parser.add_argument("--field", required=True, help="Reference-image field name, e.g. image_urls")
    parser.add_argument("--field-shape", choices=["list", "string"], default="list")
    parser.add_argument("--photo-path", required=True, help="Object path in the product-photos bucket")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--param", action="append", default=[], help="Extra body field, key=value (repeatable)")
    parser.add_argument("--yes", action="store_true", help="Skip the confirmation before the paid generation")
    args = parser.parse_args()

    model_path = args.model_path.strip("/")
    signed_url = storage.create_signed_url(storage.PRODUCT_PHOTO_BUCKET, args.photo_path)
    body = build_body(args.prompt, args.field, args.field_shape, signed_url, args.param)
    print(f"Model: {model_path}\nBody:  {json.dumps(redacted(body, args.field))}")

    estimate = higgsfield._call("POST", f"/estimate/{model_path}", json=body)
    print(f"Estimate: ${estimate.get('usd')} ({estimate.get('credits')} credits)")
    without_ref = {k: v for k, v in body.items() if k != args.field}
    estimate_plain = higgsfield._call("POST", f"/estimate/{model_path}", json=without_ref)
    print(f"Estimate without the reference: ${estimate_plain.get('usd')}")
    if estimate == estimate_plain:
        print("  Same price with and without the reference: the field may be ignored. Check the field name.")

    if not args.yes and input("Run ONE paid generation now? [y/N] ").strip().lower() != "y":
        sys.exit("Stopped before generating. Nothing was charged.")

    job = higgsfield._call(
        "POST", f"/{model_path}", json=body, headers={"Idempotency-Key": uuid.uuid4().hex}
    )
    request_id = job.get("request_id")
    print(f"Submitted request {request_id}")
    status = wait_for_result(request_id)
    if status["status"] != "completed":
        sys.exit(f"Generation ended as {status['status']}: {status.get('error')} (failed and nsfw are not charged)")

    OUTPUT_DIR.mkdir(exist_ok=True)
    for index, image in enumerate(status.get("images") or [], 1):
        response = httpx.get(image["url"], timeout=60, follow_redirects=True)
        response.raise_for_status()
        out = OUTPUT_DIR / f"{request_id}_{index}.png"
        out.write_bytes(response.content)
        print(f"Saved {out}")
    reference = httpx.get(signed_url, timeout=60)
    reference.raise_for_status()
    (OUTPUT_DIR / f"{request_id}_reference.jpg").write_bytes(reference.content)
    print(f"Saved the reference photo next to it for comparison: {OUTPUT_DIR}/{request_id}_reference.jpg")


if __name__ == "__main__":
    main()
