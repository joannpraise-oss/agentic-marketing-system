import io
import logging
from dataclasses import dataclass

from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

MAX_BYTES = 10 * 1024 * 1024  # uploads must be under 10 MB
MIN_SHORT_SIDE_PX = 1000
MAX_PIXELS = 40_000_000  # refuse decompression bombs before decoding
STORED_MAX_LONG_SIDE_PX = 2048  # keeps stored files well under Supabase's 6 MB standard-upload guidance
JPEG_QUALITY = 90
ALLOWED_FORMATS = {"JPEG", "PNG"}


class ProductPhotoError(Exception):
    """The upload is not a usable product photo. The message is safe to show the caller."""


@dataclass
class PreparedPhoto:
    data: bytes  # JPEG, no EXIF (so no GPS or camera data)
    width: int
    height: int


def prepare_product_photo(data: bytes) -> PreparedPhoto:
    """Check a product photo (JPG/PNG, under 10 MB, short side >= 1000px) and re-encode it for storage."""
    if not data:
        raise ProductPhotoError("The file is empty.")
    if len(data) >= MAX_BYTES:
        raise ProductPhotoError("The photo must be under 10 MB.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            image_format = probe.format
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError):
        raise ProductPhotoError("The file is not a readable JPG or PNG image.") from None
    # Judge the type by the bytes, not the file name or declared content type.
    if image_format not in ALLOWED_FORMATS:
        raise ProductPhotoError(f"The photo must be a JPG or PNG, not {image_format or 'an unknown format'}.")

    try:
        image = Image.open(io.BytesIO(data))  # verify() leaves the first handle unusable
        if image.width * image.height > MAX_PIXELS:
            raise ProductPhotoError("The photo has too many pixels to process.")
        image = ImageOps.exif_transpose(image)  # apply camera rotation before measuring
        short_side = min(image.size)
        if short_side < MIN_SHORT_SIDE_PX:
            raise ProductPhotoError(
                f"The photo is {image.width}x{image.height}px; its short side must be at least {MIN_SHORT_SIDE_PX}px."
            )
        image = _to_rgb(image)
        image.thumbnail((STORED_MAX_LONG_SIDE_PX, STORED_MAX_LONG_SIDE_PX), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)  # no exif= argument: metadata dropped
    except ProductPhotoError:
        raise
    except (OSError, ValueError, Image.DecompressionBombError):
        raise ProductPhotoError("The file is not a readable JPG or PNG image.") from None
    return PreparedPhoto(data=out.getvalue(), width=image.width, height=image.height)


def _to_rgb(image: Image.Image) -> Image.Image:
    # JPEG has no transparency: put transparent PNGs on white rather than black.
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")
