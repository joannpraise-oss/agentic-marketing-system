import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from decimal import ROUND_HALF_UP, Decimal
from typing import Callable

from fastapi import APIRouter, Depends, File, UploadFile
from sqlmodel import Session

from core.placeholders import asset_texts, find_placeholders
from core.spend import check_spend_caps
from db.database import get_session
from db.models import Asset
from schemas.creative import CreativeRequest
from services import creative, higgsfield, product_photo, storage

router = APIRouter()
logger = logging.getLogger(__name__)

EXTENSIONS = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
COST_PRECISION = Decimal("0.0001")  # Asset.cost_usd is Numeric(10, 4)


def _failure(exc: Exception) -> tuple[str, str]:
    """Code and caller-safe reason for one failed image (our own exception messages carry no secrets)."""
    if isinstance(exc, higgsfield.HiggsfieldAuthError):
        return "internal_error", "Something went wrong on our side."
    if isinstance(exc, higgsfield.HiggsfieldTimeout):
        return "image_timeout", str(exc)
    if isinstance(exc, higgsfield.HiggsfieldError):
        return "image_provider_error", str(exc)
    return "storage_error", str(exc)


def _upload(image: higgsfield.GeneratedImage) -> tuple[str | None, Exception | None]:
    """Returns the upload error instead of raising, because the image is already paid for."""
    try:
        return storage.upload_asset(image.data, image.content_type, EXTENSIONS.get(image.content_type, "")), None
    except storage.StorageError as exc:
        return None, exc


def _product_image_generator(photo_path: str) -> Callable[[str], higgsfield.GeneratedImage]:
    try:
        reference_url = storage.create_signed_url(storage.PRODUCT_PHOTO_BUCKET, photo_path)
    except storage.StorageObjectNotFound:
        raise product_photo.ProductPhotoError("We can't find that product photo. Please upload it again.") from None
    return lambda prompt: higgsfield.generate_product_image(prompt, reference_url)


# Plain `def` (not async): every step blocks, so FastAPI runs this in its threadpool.
@router.post("/creative/generate")
def generate_creatives(data: CreativeRequest, session: Session = Depends(get_session)):
    has_photo = data.product_photo_path is not None
    image_cost = higgsfield.estimate_product_image_cost() if has_photo else higgsfield.estimate_cost()
    # Nothing is spent before this check: planning is covered by a reserve, images by the estimate.
    count = creative.creative_count(data)
    check_spend_caps(session, image_cost * count + creative.PLANNING_RESERVE_USD)

    # Signed before planning, so a bad photo path costs nothing.
    generate = _product_image_generator(data.product_photo_path) if has_photo else higgsfield.generate_image
    plan, planning_cost = creative.plan_creatives(data)
    if has_photo:
        for item in plan.items:  # changed here, so the saved asset keeps the prompt actually sent
            item.prompt = creative.with_product_label(item.prompt, data.brand.brand_name)
    test_group_id = uuid.uuid4().hex if data.creative_test else None
    test_mode = ("varied_format" if data.vary_format else "fixed_format") if data.creative_test else None

    # Generate in parallel (each job waits on Higgsfield), then upload one at a time:
    # parallel uploads were the suspect for intermittent ReadErrors from Supabase Storage.
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [pool.submit(generate, item.prompt) for item in plan.items]
    outcomes = []
    unexpected = None
    for item, future in zip(plan.items, futures):
        try:
            image = future.result()
        except Exception as exc:  # generation failed: not charged, nothing to save
            outcomes.append((item, None, None, exc))
            if not isinstance(exc, higgsfield.HiggsfieldError):
                unexpected = unexpected or exc  # a bug, not a provider failure: raise after saving paid work
            continue
        try:
            outcomes.append((item, image, *_upload(image)))
        except Exception as exc:
            outcomes.append((item, image, None, exc))
            unexpected = unexpected or exc

    # Save every paid image (even one whose upload failed) so the daily cap sees the real spend.
    # Planning cost is split evenly across the assets we save.
    paid = [o for o in outcomes if o[1] is not None]
    planning_share = planning_cost / len(paid) if paid else Decimal("0")
    assets = []
    for item, image, url, _ in paid:
        asset = Asset(
            campaign_brief=data.brief,
            kind="image",
            prompt=item.prompt,
            model=higgsfield.PRODUCT_MODEL if has_photo else higgsfield.MODEL,
            provider=higgsfield.PROVIDER,
            file_url=url,
            caption=item.caption,
            alt_text=item.alt_text,
            faq=[faq.model_dump() for faq in item.faq],
            format=item.format,
            offer_type=data.offer_type,
            angle=item.angle,
            text_rights_confirmed=True if data.source_text else None,
            text_choice=data.text_choice,
            text_mode=data.text_mode if data.text_choice != "none" else None,
            on_image_text=item.on_image_text,
            test_group_id=test_group_id,
            test_mode=test_mode,
            core_message=item.core_message,
            status="draft",
            cost_usd=(image_cost + planning_share).quantize(COST_PRECISION, rounding=ROUND_HALF_UP),
        )
        session.add(asset)
        assets.append(asset)
    if assets:
        session.commit()
        for asset in assets:
            session.refresh(asset)
        logger.info("Saved %s creative drafts: %s", len(assets), [a.id for a in assets])

    if unexpected is not None:
        raise unexpected

    # Every paid image is saved above; only one with a stored file counts as a usable draft.
    saved = iter(assets)
    drafts, failed = [], []
    for item, image, url, error in outcomes:
        asset = next(saved) if image is not None else None
        if error is None:
            drafts.append(asset)
            continue
        code, reason = _failure(error)
        failed.append({
            "format": item.format, "code": code, "reason": reason,
            "asset_id": asset.id if asset is not None else None,
        })

    if failed:
        logger.warning("Creative run: %s of %s image(s) failed: %s", len(failed), len(outcomes), failed)
    if not drafts:
        # Nothing usable: mapped to 504 / 502 by core.errors. Paid images are already saved.
        raise next(o[3] for o in outcomes if o[3] is not None)

    return {
        "test_group_id": test_group_id,
        "test_mode": test_mode,
        "format_rationale": plan.format_rationale,
        "assets": [_asset_out(asset) for asset in drafts],
        "failed_count": len(failed),
        "failed": failed,
    }


def _asset_out(asset: Asset) -> dict:
    # needs_input is derived from the text, so it can never disagree with what's in the asset.
    return {
        **asset.model_dump(mode="json"),
        "needs_input": find_placeholders(asset_texts(asset.caption, asset.alt_text, asset.faq)),
    }


# Plain `def`: reading the upload, Pillow and the storage call all block.
@router.post("/creative/product-photo")
def upload_product_photo(file: UploadFile = File(...)):
    """Convert and store a product photo in the private bucket; returns its path for /creative/generate.

    Fixable problems (format, file size) are fixed; quality concerns come back as warnings, not errors.
    """
    data = file.file.read(product_photo.MAX_UPLOAD_BYTES)  # stop reading at the limit
    if len(data) >= product_photo.MAX_UPLOAD_BYTES:
        raise product_photo.ProductPhotoError("The file is over 50 MB. Please upload a smaller photo.")
    photo = product_photo.prepare_product_photo(data)
    path = storage.upload_product_photo(photo.data)
    logger.info(
        "Stored product photo %s (%sx%s, %.1f KB, warnings=%s)",
        path, photo.width, photo.height, len(photo.data) / 1024, [w.code for w in photo.warnings],
    )
    return {
        "product_photo_path": path,
        "width": photo.width,
        "height": photo.height,
        "size_kb": round(len(photo.data) / 1024, 1),
        "warnings": [{"code": w.code, "message": w.message} for w in photo.warnings],
    }
