import io
import logging
from dataclasses import dataclass, field

import pillow_heif
from PIL import Image, ImageOps, UnidentifiedImageError

logger = logging.getLogger(__name__)

pillow_heif.register_heif_opener()  # iPhone photos (HEIC/HEIF)

# Fix and warn: anything Pillow can read as an image is accepted, converted to JPEG and
# shrunk for storage. Only non-images (and files too big to handle safely) are rejected.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # memory guard, not a quality rule; big photos are shrunk
MIN_SHORT_SIDE_PX = 768  # below this we still accept, with a small_image warning
MAX_PIXELS = 120_000_000  # refuse decompression bombs before decoding; fits 108MP phone photos
STORED_MAX_LONG_SIDE_PX = 2048  # keeps stored files well under Supabase's 6 MB standard-upload guidance
JPEG_QUALITY = 90


class ProductPhotoError(Exception):
    """The upload is not a usable image. The message is safe to show the caller."""


@dataclass
class PhotoWarning:
    """A tip about the photo: it was accepted, but could give better results."""
    code: str
    message: str


@dataclass
class PreparedPhoto:
    data: bytes  # JPEG, no EXIF (so no GPS or camera data)
    width: int
    height: int
    warnings: list[PhotoWarning] = field(default_factory=list)


def prepare_product_photo(data: bytes) -> PreparedPhoto:
    """Turn any readable image into a stored JPEG (max 2048px long side), with warnings instead of rejections."""
    if not data:
        raise ProductPhotoError("The file is empty.")
    if len(data) >= MAX_UPLOAD_BYTES:
        raise ProductPhotoError("The file is over 50 MB. Please upload a smaller photo.")
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
        raise ProductPhotoError("This file isn't an image we can read. Please upload a photo (JPG, PNG, HEIC or WebP).") from None

    try:
        image = Image.open(io.BytesIO(data))  # verify() leaves the first handle unusable
        if image.width * image.height > MAX_PIXELS:
            raise ProductPhotoError("The photo has too many pixels to process. Please upload a smaller photo.")
        original_size = _upright_size(image)  # measure before draft() scales a JPEG down
        # JPEGs can decode at reduced scale: saves memory on big phone photos.
        image.draft("RGB", (STORED_MAX_LONG_SIDE_PX, STORED_MAX_LONG_SIDE_PX))
        image = ImageOps.exif_transpose(image)
        image = _to_rgb(image)
        image.thumbnail((STORED_MAX_LONG_SIDE_PX, STORED_MAX_LONG_SIDE_PX), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        image.save(out, "JPEG", quality=JPEG_QUALITY, optimize=True)  # no exif= argument: metadata dropped
    except ProductPhotoError:
        raise
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError):
        raise ProductPhotoError("This file isn't an image we can read. Please upload a photo (JPG, PNG, HEIC or WebP).") from None
    return PreparedPhoto(data=out.getvalue(), width=image.width, height=image.height, warnings=_warnings(original_size))


def _upright_size(image: Image.Image) -> tuple[int, int]:
    # Size as the camera meant it: some orientations store the photo sideways.
    width, height = image.size
    if image.getexif().get(0x0112) in (5, 6, 7, 8):
        width, height = height, width
    return width, height


def _warnings(size: tuple[int, int]) -> list[PhotoWarning]:
    width, height = size
    warnings = []
    if min(width, height) < MIN_SHORT_SIDE_PX:
        warnings.append(PhotoWarning(
            "small_image",
            f"This photo is {width}x{height}px, so your images may look a little soft. "
            f"For the sharpest results, use a photo that's at least {MIN_SHORT_SIDE_PX}px on its shortest side.",
        ))
    return warnings


def _to_rgb(image: Image.Image) -> Image.Image:
    # JPEG has no transparency: put transparent PNGs on white rather than black.
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.getchannel("A"))
        return background
    return image.convert("RGB")
