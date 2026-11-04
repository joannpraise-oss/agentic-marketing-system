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
    """Fake vision model: set `text` to what it reads, or `error` to make the call fail.

    `texts` (one per call, None for a failed call) overrides `text` while it lasts.
    """
    state = SimpleNamespace(calls=[], text="JB\nJ'ADUNÉ\nBEAUTY", error=None, texts=[])

    def create_with_completion(**kwargs):
        state.calls.append(kwargs)
        if state.texts:
            text = state.texts.pop(0)
            if text is None:
                raise RuntimeError("vision down")
            usage = SimpleNamespace(prompt_tokens=25_000, completion_tokens=20)
            return label_check.LabelReading(label_text=text), SimpleNamespace(usage=usage)
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


# --- the second read ---

def test_a_passing_first_read_is_not_repeated(vision):
    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert result.ok and len(vision.calls) == 1
    assert not result.read_twice and result.reads_logged == "JB\nJ'ADUNÉ\nBEAUTY"


def test_a_misread_rescued_by_the_second_read_passes(vision):
    vision.texts = ["JB\nJADUNÉ\nBEAUTY", "JB\nJ'ADUNÉ\nBEAUTY"]  # live: #27's apostrophe dropped once

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert result.ok and result.mismatched == [] and len(vision.calls) == 2
    assert result.reads_logged == "JB\nJADUNÉ\nBEAUTY\n--- second read ---\nJB\nJ'ADUNÉ\nBEAUTY"
    assert result.cost_usd == 2 * Decimal("0.003762")  # both reads are paid for


def test_flagged_only_when_both_reads_disagree(vision):
    reference = "JB J’aduné BEAUTY PEPTIDE INFUSED"
    vision.texts = ["JB\nJ'ADUNÉ\nBEAUTY\nPEPTIDE-INFUSED", "JB J'ADUNÉ BEAUTY PEPTIDE-INFUSED"]

    result = label_check.check_label(b"png", "image/png", "J’aduné", reference)

    assert not result.ok
    assert result.mismatched == ["missing: infused", "missing: peptide", "extra: peptide-infused"]
    assert result.text_read == "JB\nJ'ADUNÉ\nBEAUTY\nPEPTIDE-INFUSED"
    assert result.second_read == "JB J'ADUNÉ BEAUTY PEPTIDE-INFUSED"


def test_a_failed_first_read_is_retried(vision):
    vision.texts = [None, "J'ADUNÉ"]

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert result.ok and result.reads_logged == "(read failed)\n--- second read ---\nJ'ADUNÉ"


def test_a_failed_second_read_keeps_the_flag_and_the_first_reads_differences(vision):
    vision.texts = ["TADUNE", None]

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert not result.ok and result.mismatched == ["brand: expected 'J’aduné'"]
    assert result.reads_logged == "TADUNE\n--- second read ---\n(read failed)"


def test_a_failed_check_flags_the_image_instead_of_raising(vision):
    vision.error = RuntimeError("vision down")

    result = label_check.check_label(b"png", "image/png", "J’aduné")

    assert result == label_check.LabelCheck(
        ok=False, text_read=None, cost_usd=Decimal("0"), read_twice=True, outcome="failed"
    )
    assert len(vision.calls) == 2 and result.reads_logged is None


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


def test_label_differences_ignore_order_case_apostrophe_style_and_line_breaks():
    reference = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"
    read = "JB\nJ'ADUNÉ\nBEAUTY\nhydrating hair mist\nPEPTIDE INFUSED\n8 fl oz (237 mL)"

    assert label_check.label_differences(reference, read) == []


@pytest.mark.parametrize("read, differences", [
    ("8 fl oz (237 mL).", ["missing: ml)", "extra: ml)."]),  # punctuation
    ("8 fl oz (236 mL)", ["missing: (237", "extra: (236"]),  # a number
    ("8 fl oz / 237 mL", ["missing: (237", "missing: ml)", "extra: /", "extra: 237", "extra: ml"]),
    ("8 fl oz (237 mL) PEPTIDE-INFUSED", ["extra: peptide-infused"]),  # an invented word
])
def test_any_readable_difference_counts(read, differences):
    assert label_check.label_differences("8 fl oz (237 mL)", read) == differences


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


# --- three outcomes: match, unreadable (passes), mismatch (flagged) ---

REFERENCE = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"


@pytest.mark.parametrize("read, outcome", [
    ("JB\nJ'ADUNÉ\nBEAUTY\nPEPTIDE INFUSED\nhydrating hair mist\n8 fl oz (237 mL)", "match"),
    # small print too small to read: passes, isn't flagged
    ("JB\nJ'ADUNÉ\nBEAUTY\n[unreadable]\nhydrating hair mist\n[unreadable]", "unreadable"),
    ("JB\nJ'ADUNÉ\nBEAUTY\nPEP??DE INFUSED\nhydrating hair mist\n8 fl oz (2?7 mL)", "unreadable"),
    # readable but different: flagged, even next to unreadable text
    ("JB\nJ'ADUNÉ\nBEAUTY\n[unreadable]\nhydrating hair mist\n8 fl oz (236 mL)", "mismatch"),
    ("JB\nJ'ADUNÉ\nBEAUTY\nhydrating hair mist\n8 fl oz (237 mL)", "mismatch"),  # badge gone, nothing unreadable
    # the brand must always be readable and exact
    ("JB\nJ'ADUN?\nBEAUTY\n[unreadable]", "mismatch"),
    ("JB\n[unreadable]\nBEAUTY\nPEPTIDE INFUSED\nhydrating hair mist\n8 fl oz (237 mL)", "mismatch"),
    ("JB\nJADUNÉ\nBEAUTY\nPEPTIDE INFUSED\nhydrating hair mist\n8 fl oz (237 mL)", "mismatch"),
])
def test_assess_has_three_outcomes(read, outcome):
    assert label_check.assess(read, "J’aduné", REFERENCE)[0] == outcome


def test_unreadable_small_print_passes_without_a_second_read(vision):
    vision.text = "JB\nJ'ADUNÉ\nBEAUTY\n[unreadable]\nhydrating hair mist\n[unreadable]"

    result = label_check.check_label(b"png", "image/png", "J’aduné", REFERENCE)

    assert result.ok and result.outcome == "unreadable" and result.mismatched == []
    assert len(vision.calls) == 1


def test_mismatch_then_unreadable_passes_as_unreadable(vision):
    vision.texts = ["JB\nJ'ADUNÉ\nBEAUTY\n8 fl oz (236 mL)", "JB\nJ'ADUNÉ\nBEAUTY\n[unreadable]"]

    result = label_check.check_label(b"png", "image/png", "J’aduné", REFERENCE)

    assert result.ok and result.outcome == "unreadable"


def test_readable_difference_in_both_reads_is_a_mismatch(vision):
    vision.text = "JB\nJ'ADUNÉ\nBEAUTY\n[unreadable]\nhydrating hair mist\n8 fl oz (236 mL)"

    result = label_check.check_label(b"png", "image/png", "J’aduné", REFERENCE)

    assert not result.ok and result.outcome == "mismatch"
    assert result.mismatched == ["extra: (236"]


def test_reader_is_asked_to_mark_unreadable_text(vision):
    label_check.check_label(b"png", "image/png", "J’aduné")

    assert "[unreadable]" in vision.calls[0]["messages"][0]["content"][0]["text"]


def test_read_label_returns_none_instead_of_raising(vision):
    vision.error = RuntimeError("vision down")

    assert label_check.read_label(b"png", "image/png") == (None, Decimal("0"))
