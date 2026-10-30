import io
import os

import pytest
from PIL import Image

from services import product_photo, storage


def image_bytes(size, fmt="PNG", mode="RGB", color=(200, 40, 40), **save_kwargs):
    out = io.BytesIO()
    Image.new(mode, size, color if mode != "RGBA" else (*color, 0)).save(out, fmt, **save_kwargs)
    return out.getvalue()


def with_exif(size, orientation=None, gps=False):
    image = Image.new("RGB", size, (10, 120, 60))
    exif = Image.Exif()
    exif[0x010F] = "PhoneMaker"  # camera make
    if orientation:
        exif[0x0112] = orientation
    if gps:
        exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0)}
    out = io.BytesIO()
    image.save(out, "JPEG", exif=exif)
    return out.getvalue()


# --- checks ---

@pytest.mark.parametrize("fmt", ["PNG", "JPEG", "WEBP", "HEIF", "GIF"])
def test_any_image_format_is_converted_to_jpeg(fmt):
    photo = product_photo.prepare_product_photo(image_bytes((1200, 1000), fmt))

    stored = Image.open(io.BytesIO(photo.data))
    assert stored.format == "JPEG"
    assert (photo.width, photo.height) == stored.size == (1200, 1000)
    assert photo.warnings == []


def test_exif_and_gps_are_stripped():
    photo = product_photo.prepare_product_photo(with_exif((1500, 1200), gps=True))

    assert not Image.open(io.BytesIO(photo.data)).getexif()


def test_camera_rotation_applied_before_measuring():
    # Stored 1600x1000 with "rotate 90" orientation: really a 1000x1600 portrait.
    photo = product_photo.prepare_product_photo(with_exif((1600, 1000), orientation=6))

    assert (photo.width, photo.height) == (1000, 1600)


def test_large_photos_are_scaled_to_2048_long_side():
    photo = product_photo.prepare_product_photo(image_bytes((4000, 3000), "JPEG"))

    assert (photo.width, photo.height) == (2048, 1536)


def test_transparent_png_goes_on_white():
    photo = product_photo.prepare_product_photo(image_bytes((1000, 1000), "PNG", mode="RGBA"))

    assert Image.open(io.BytesIO(photo.data)).getpixel((10, 10)) == pytest.approx((255, 255, 255), abs=2)


def test_48mp_phone_jpeg_is_accepted_and_shrunk():
    # 48.8M pixels: over the old 40M pixel guard, which rejected recent iPhone photos.
    photo = product_photo.prepare_product_photo(image_bytes((8064, 6048), "JPEG"))

    assert max(photo.width, photo.height) == 2048
    assert photo.warnings == []


@pytest.mark.parametrize("size, fmt, shown", [
    ((999, 1600), "PNG", "999x1600px"),
    ((1600, 800), "JPEG", "1600x800px"),
    ((400, 300), "WEBP", "400x300px"),
])
def test_small_photo_is_accepted_with_warning(size, fmt, shown):
    photo = product_photo.prepare_product_photo(image_bytes(size, fmt))

    assert (photo.width, photo.height) == size  # never upscaled
    assert [w.code for w in photo.warnings] == ["small_image"]
    assert shown in photo.warnings[0].message and "1000px" in photo.warnings[0].message


def test_small_image_warning_uses_upright_size():
    # Stored 1600x900 with "rotate 90": really 900x1600, short side 900.
    photo = product_photo.prepare_product_photo(with_exif((1600, 900), orientation=6))

    assert "900x1600px" in photo.warnings[0].message


@pytest.mark.parametrize("data, message", [
    (b"", "empty"),
    (b"not an image at all", "isn't an image we can read"),
    (b"%PDF-1.7\n" + b"x" * 2000, "isn't an image we can read"),
    (image_bytes((1200, 1200), "PNG")[:200], "isn't an image we can read"),  # truncated
    (b"x" * product_photo.MAX_UPLOAD_BYTES, "over 50 MB"),
])
def test_non_images_are_rejected(data, message):
    with pytest.raises(product_photo.ProductPhotoError, match=message):
        product_photo.prepare_product_photo(data)


def test_too_many_pixels_rejected_before_decoding(monkeypatch):
    monkeypatch.setattr(product_photo, "MAX_PIXELS", 1_000_000)
    with pytest.raises(product_photo.ProductPhotoError, match="too many pixels"):
        product_photo.prepare_product_photo(image_bytes((1200, 1200)))


# --- endpoint ---

@pytest.fixture
def stored(monkeypatch):
    uploads = []

    def fake_upload(data, **kwargs):
        uploads.append(data)
        return "abc123.jpg"

    monkeypatch.setattr(storage, "upload_product_photo", fake_upload)
    return uploads


def post_photo(client, data, name="photo.png", content_type="image/png"):
    return client.post("/creative/product-photo", files={"file": (name, data, content_type)})


def test_upload_stores_photo_and_returns_private_path(client, stored):
    resp = post_photo(client, image_bytes((1200, 1000)))

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["product_photo_path"] == "abc123.jpg"
    assert (body["width"], body["height"]) == (1200, 1000)
    assert "url" not in " ".join(body)  # private bucket: no public URL is handed out
    assert body["warnings"] == []
    assert Image.open(io.BytesIO(stored[0])).format == "JPEG"


def test_heic_upload_is_stored_as_jpeg(client, stored):
    resp = post_photo(client, image_bytes((1500, 1200), "HEIF"), name="IMG_0001.HEIC", content_type="image/heic")

    assert resp.status_code == 200, resp.text
    assert Image.open(io.BytesIO(stored[0])).format == "JPEG"


def test_upload_over_old_10mb_limit_is_shrunk(client, stored):
    # Random noise barely compresses, so this PNG is well over 10 MB.
    noisy = Image.frombytes("RGB", (3000, 2000), os.urandom(3000 * 2000 * 3))
    out = io.BytesIO()
    noisy.save(out, "PNG", compress_level=0)
    assert len(out.getvalue()) > 10 * 1024 * 1024

    resp = post_photo(client, out.getvalue(), name="big.png")

    assert resp.status_code == 200, resp.text
    assert (resp.json()["width"], resp.json()["height"]) == (2048, 1365)


def test_low_resolution_upload_is_accepted_with_warning(client, stored):
    resp = post_photo(client, image_bytes((800, 1200)))

    assert resp.status_code == 200, resp.text
    [warning] = resp.json()["warnings"]
    assert warning["code"] == "small_image"
    assert "800x1200px" in warning["message"]
    assert len(stored) == 1


def test_non_image_is_400_even_with_image_content_type(client, stored):
    resp = post_photo(client, b"<html>not a photo</html>", name="photo.png", content_type="image/png")

    assert resp.status_code == 400
    body = resp.json()
    assert body["code"] == "invalid_product_photo"
    assert "isn't an image" in body["detail"]
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert stored == []


def test_upload_over_memory_guard_is_400(client, stored):
    resp = post_photo(client, b"x" * (product_photo.MAX_UPLOAD_BYTES + 10), name="big.jpg", content_type="image/jpeg")

    assert resp.status_code == 400
    assert "over 50 MB" in resp.json()["detail"]
    assert stored == []


def test_missing_file_is_400(client, stored):
    resp = client.post("/creative/product-photo")

    assert resp.status_code == 400
    assert resp.json()["code"] == "invalid_input"


def test_storage_failure_is_502(client, monkeypatch):
    def failing(data, **kwargs):
        raise storage.StorageError("Storage upload failed with status 500.")

    monkeypatch.setattr(storage, "upload_product_photo", failing)

    resp = post_photo(client, image_bytes((1200, 1000)))

    assert resp.status_code == 502
    assert resp.json()["code"] == "storage_error"
