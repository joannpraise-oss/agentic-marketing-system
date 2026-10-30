import importlib.util
import io
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "try_reference_model.py"
spec = importlib.util.spec_from_file_location("try_reference_model", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

SIGNED = "https://proj.supabase.co/storage/v1/object/sign/product-photos/clean.jpg?token=SECRET-TOKEN"
ARGS = [
    "try_reference_model.py", "--model-path", "/vendor/model/edit", "--field", "image_urls",
    "--photo-path", "p.jpg", "--prompt", "Studio hero shot of this product",
    "--param", "aspect_ratio=1:1", "--param", "num_images=1",
]


def test_build_body_list_and_string_shapes():
    body = script.build_body("p", "image_urls", "list", SIGNED, ["aspect_ratio=1:1", "num_images=2", "x=true"])
    assert body == {"prompt": "p", "aspect_ratio": "1:1", "num_images": 2, "x": True, "image_urls": [SIGNED]}
    assert script.build_body("p", "image_url", "string", SIGNED, [])["image_url"] == SIGNED


def phone_photo_with_gps(size=(1500, 1200)):
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x8825] = {1: "N", 2: (51.0, 30.0, 0.0)}
    out = io.BytesIO()
    Image.new("RGB", size, (10, 120, 60)).save(out, "JPEG", exif=exif)
    return out.getvalue()


@pytest.fixture
def fake_storage(monkeypatch):
    """The bucket holds a raw phone photo; records what the script uploads, signs and deletes."""
    bucket = {"p.jpg": phone_photo_with_gps()}
    log = {"uploaded": [], "signed": [], "deleted": []}

    def upload(data, **kwargs):
        bucket["clean.jpg"] = data
        log["uploaded"].append(data)
        return "clean.jpg"

    def sign(bucket_name, path, *a, **k):
        log["signed"].append(path)
        return SIGNED

    monkeypatch.setattr(script.storage, "download_product_photo", lambda path: bucket[path])
    monkeypatch.setattr(script.storage, "upload_product_photo", upload)
    monkeypatch.setattr(script.storage, "create_signed_url", sign)
    monkeypatch.setattr(script.storage, "delete_product_photo", log["deleted"].append)
    return bucket, log


@pytest.fixture
def fake_api(monkeypatch, fake_storage):
    calls = []

    def fake_call(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path.startswith("/estimate/"):
            has_ref = "image_urls" in kwargs["json"]
            return {"usd": "0.039" if has_ref else "0.030", "credits": "0.6"}
        return {"request_id": "r1", "status": "queued"}

    monkeypatch.setattr(script.higgsfield, "_call", fake_call)
    monkeypatch.setattr("sys.argv", ARGS)
    return calls


def test_shows_price_and_stops_without_confirmation(fake_api, monkeypatch, capsys):
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    with pytest.raises(SystemExit, match="Nothing was charged"):
        script.main()

    out = capsys.readouterr().out
    assert "Estimate: $0.039" in out and "Estimate without the reference: $0.030" in out
    assert [c[1] for c in fake_api] == ["/estimate/vendor/model/edit"] * 2  # nothing submitted
    assert fake_api[0][2]["json"]["image_urls"] == [SIGNED]
    assert "SECRET-TOKEN" not in out


def test_warns_when_reference_does_not_change_the_price(fake_api, monkeypatch, capsys):
    monkeypatch.setattr(script.higgsfield, "_call", lambda method, path, **kw: {"usd": "0.006", "credits": "0.09"})
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    with pytest.raises(SystemExit):
        script.main()

    assert "the field may be ignored" in capsys.readouterr().out


def test_only_a_cleaned_copy_is_signed_and_it_is_deleted_after(fake_api, fake_storage, monkeypatch):
    bucket, log = fake_storage
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    with pytest.raises(SystemExit):
        script.main()

    assert log["signed"] == ["clean.jpg"]  # never the raw original
    [uploaded] = log["uploaded"]
    cleaned = Image.open(io.BytesIO(uploaded))
    assert cleaned.format == "JPEG" and not cleaned.getexif()  # GPS and camera data gone
    assert log["deleted"] == ["clean.jpg"]
    assert "p.jpg" in bucket  # the original is left alone


def test_temp_copy_is_deleted_even_when_the_api_fails(fake_api, fake_storage, monkeypatch):
    _, log = fake_storage

    def failing(method, path, **kwargs):
        raise script.higgsfield.HiggsfieldError("Higgsfield returned status 500.")

    monkeypatch.setattr(script.higgsfield, "_call", failing)

    with pytest.raises(script.higgsfield.HiggsfieldError):
        script.main()

    assert log["deleted"] == ["clean.jpg"]


def test_small_photo_warning_is_shown(fake_api, fake_storage, monkeypatch, capsys):
    bucket, _ = fake_storage
    bucket["p.jpg"] = phone_photo_with_gps((360, 480))
    monkeypatch.setattr("builtins.input", lambda prompt: "n")

    with pytest.raises(SystemExit):
        script.main()

    assert "Warning (small_image)" in capsys.readouterr().out


def test_non_image_stops_before_anything_is_uploaded_or_sent(fake_api, fake_storage):
    bucket, log = fake_storage
    bucket["p.jpg"] = b"not an image"

    with pytest.raises(SystemExit, match="isn't an image"):
        script.main()

    assert log["uploaded"] == [] and fake_api == []
