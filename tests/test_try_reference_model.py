import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "try_reference_model.py"
spec = importlib.util.spec_from_file_location("try_reference_model", SCRIPT)
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

SIGNED = "https://proj.supabase.co/storage/v1/object/sign/product-photos/p.jpg?token=SECRET-TOKEN"
ARGS = [
    "try_reference_model.py", "--model-path", "/vendor/model/edit", "--field", "image_urls",
    "--photo-path", "p.jpg", "--prompt", "Studio hero shot of this product",
    "--param", "aspect_ratio=1:1", "--param", "num_images=1",
]


def test_build_body_list_and_string_shapes():
    body = script.build_body("p", "image_urls", "list", SIGNED, ["aspect_ratio=1:1", "num_images=2", "x=true"])
    assert body == {"prompt": "p", "aspect_ratio": "1:1", "num_images": 2, "x": True, "image_urls": [SIGNED]}
    assert script.build_body("p", "image_url", "string", SIGNED, [])["image_url"] == SIGNED


@pytest.fixture
def fake_api(monkeypatch):
    calls = []

    def fake_call(method, path, **kwargs):
        calls.append((method, path, kwargs))
        if path.startswith("/estimate/"):
            has_ref = "image_urls" in kwargs["json"]
            return {"usd": "0.039" if has_ref else "0.030", "credits": "0.6"}
        return {"request_id": "r1", "status": "queued"}

    monkeypatch.setattr(script.storage, "create_signed_url", lambda bucket, path, *a, **k: SIGNED)
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
