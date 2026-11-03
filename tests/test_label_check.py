import json
from decimal import Decimal
from types import SimpleNamespace

import pytest

from core.placeholders import needs_input
from services import creative, label_check


@pytest.mark.parametrize("label_text, ok", [
    ("J'ADUNÉ", True),  # case and apostrophe style don't matter
    ("JB\nJ’ADUNÉ\nBEAUTY\nPEPTIDE INFUSED", True),  # the brand anywhere on the label, as whole words
    ("TADUNE", False),
    ("JADUNÉ", False),  # missing apostrophe
    ("J'ADUNE", False),  # missing accent
    ("J'ADUN?", False),
    ("J'ADUNÉS", False),  # not a whole word
    ("HYDRATING HAIR MIST", False),  # brand not legible
    ("", False),
])
def test_brand_matches(label_text, ok):
    assert label_check.brand_matches(label_text, "J’aduné") is ok


def test_multi_word_brand_must_appear_in_order():
    assert label_check.brand_matches("GLOW  CO\nserum", "Glow Co")
    assert not label_check.brand_matches("CO GLOW", "Glow Co")


@pytest.fixture
def vision(monkeypatch):
    """Fake vision model: set `text` to what it reads, or `error` to make the call fail."""
    state = SimpleNamespace(calls=[], text="JB\nJ'ADUNÉ\nBEAUTY", error=None)

    def create_with_completion(**kwargs):
        state.calls.append(kwargs)
        if state.error:
            raise state.error
        usage = SimpleNamespace(prompt_tokens=25_000, completion_tokens=20)
        return label_check.LabelReading(label_text=state.text), SimpleNamespace(usage=usage)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion)))
    monkeypatch.setattr(creative, "client", client)
    return state


def test_check_sends_the_image_and_never_the_expected_name(vision):
    result = label_check.check_label(b"png-bytes", "image/png", "J’aduné")

    assert result.ok and result.text_read == "JB\nJ'ADUNÉ\nBEAUTY"
    [call] = vision.calls
    _, image_part = call["messages"][0]["content"]
    assert image_part["image_url"] == {"url": "data:image/png;base64,cG5nLWJ5dGVz", "detail": "high"}
    sent = json.dumps(call["messages"], ensure_ascii=False).casefold()
    assert "aduné" not in sent  # told the name, the model "reads" it even when garbled
    # 25,000 in x $0.15/M + 20 out x $0.60/M
    assert result.cost_usd == Decimal("0.003762")


def test_mismatch_is_not_ok_and_keeps_what_was_read(vision):
    vision.text = "JB\nTADUNE"

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert not result.ok and result.text_read == "JB\nTADUNE"


def test_a_failed_check_flags_the_image_instead_of_raising(vision):
    vision.error = RuntimeError("vision down")

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert result == label_check.LabelCheck(ok=False, text_read=None, cost_usd=Decimal("0"))


@pytest.mark.parametrize("label_ok, expected", [(None, []), (True, []), (False, ["label_mismatch"])])
def test_needs_input_adds_label_mismatch_only_for_a_failed_check(label_ok, expected):
    assert needs_input("Shop now.", None, [], label_ok) == expected
    assert needs_input("Ships in [▢ time, e.g. 2 days].", None, [], label_ok) == ["[▢ time, e.g. 2 days]", *expected]


# --- the reference label from the uploaded photo ---

PHOTO_READ = "JB\nTADUNÉ\nBEAUTY\nPEPTIDE INFUSED\nhydrating hair mist\n8 fl oz (237 mL)"  # real read of f51db36…


def test_misread_brand_in_the_reference_is_replaced_with_brand_name():
    assert label_check.reference_label(PHOTO_READ, "J’aduné") == (
        "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"
    )


def test_reference_with_the_right_brand_is_kept_as_read():
    assert label_check.reference_label("JB\nJ'ADUNÉ\nBEAUTY", "J’aduné") == "JB\nJ'ADUNÉ\nBEAUTY"


def test_reference_without_anything_like_the_brand_is_left_alone():
    assert label_check.reference_label("PEPTIDE INFUSED\nhair mist", "J’aduné") == "PEPTIDE INFUSED\nhair mist"


def test_label_differences_ignore_order_case_punctuation_and_line_breaks():
    reference = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"
    read = "JB\nJ'ADUNÉ\nBEAUTY\nhydrating hair mist\nPEPTIDE INFUSED\n8 fl oz (237 mL)."

    assert label_check.label_differences(reference, read) == []


def test_label_differences_name_each_missing_and_invented_word():
    reference = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"
    read = "JB\nJ'ADUNÉ\nBEAUTY\npuré white\nhydrating hair mist\n8 fl oz (237 mL)"  # live misread of #23

    assert label_check.label_differences(reference, read) == [
        "missing: infused", "missing: peptide", "extra: puré", "extra: white",
    ]


def test_with_a_reference_any_wrong_word_fails_the_check(vision):
    vision.text = "JB\nJ'ADUNÉ\nBEAUTY\npuré white\nhydrating hair mist\n8 fl oz (237 mL)"
    reference = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"

    result = label_check.check_label(b"png", "image/png", "J’aduné", reference)

    assert not result.ok
    assert result.mismatched == ["missing: infused", "missing: peptide", "extra: puré", "extra: white"]


def test_with_a_reference_a_matching_label_passes(vision):
    vision.text = "JB\nJ'ADUNÉ\nBEAUTY\nPEPTIDE INFUSED\nhydrating hair mist\n8 fl oz (237 mL)"
    reference = label_check.reference_label(PHOTO_READ, "J’aduné")

    result = label_check.check_label(b"png", "image/png", "J’aduné", reference)

    assert result.ok and result.mismatched == []


def test_brand_is_still_checked_against_brand_name_with_a_reference(vision):
    vision.text = "JB\nTADUNÉ\nBEAUTY"

    result = label_check.check_label(b"png", "image/png", "J’aduné", "JB TADUNÉ BEAUTY")

    assert not result.ok and result.mismatched == ["brand: expected 'J’aduné'"]


def test_read_label_returns_none_instead_of_raising(vision):
    vision.error = RuntimeError("vision down")

    assert label_check.read_label(b"png", "image/png") == (None, Decimal("0"))
