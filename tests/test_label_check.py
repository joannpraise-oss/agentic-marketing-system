import base64
import io
import json
from decimal import Decimal
from types import SimpleNamespace

import pytest
from PIL import Image

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

    `texts` (one per read, None for a failed read) overrides `text` while it lasts. Reads are
    recorded in `calls`; the product-locate call answers `cells` (none by default, at no cost)
    and is recorded in `locates`.
    """
    state = SimpleNamespace(calls=[], text="JB\nJ'ADUNÉ\nBEAUTY", error=None, texts=[], locates=[],
                            cells=[], locate_tokens=0, locate_error=None)

    def create_with_completion(**kwargs):
        if kwargs["response_model"] is label_check.LabelCells:
            state.locates.append(kwargs)
            if state.error or state.locate_error:
                raise state.error or state.locate_error
            usage = SimpleNamespace(prompt_tokens=state.locate_tokens, completion_tokens=0)
            return label_check.LabelCells(cells=state.cells), SimpleNamespace(usage=usage)
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
    assert result.mismatched == ["extra: peptide-infused"]  # both reads saw the hyphen
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
    assert result.mismatched == ["extra: puré", "extra: white"]


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


# --- combining two reads that both found a problem (real reads of #30 and #31) ---

PHOTO_REFERENCE = "JB J’aduné BEAUTY PEPTIDE INFUSED hydrating hair mist 8 fl oz (237 mL)"


def test_garbled_small_print_read_differently_each_time_is_unreadable():
    first = "J.B\nJ'ADUNÉ\nBEAUTY\nPOTTO INTEIRO\nhydrating hair mist\n8 fl oz (237 mL)"
    second = "JB\nJ'ADUNÉ\nBEAUTY\nPOTION INEDITE\nhydrating hair mist\n8 fl oz (237 mL)"

    assert label_check.combine_reads(first, second, "J’aduné", PHOTO_REFERENCE) == ("unreadable", [])


def test_small_print_both_reads_skipped_is_unreadable():
    read = "JB\nJ'ADUNÉ\nHYDRATING HAIR MIST\n8 fl oz (237 ml)"

    assert label_check.combine_reads(read, read, "J’aduné", PHOTO_REFERENCE) == ("unreadable", [])


def test_the_same_wrong_word_in_both_reads_is_a_mismatch():
    first = "JB\nJ'ADUNÉ\nBEAUTY\nhydrating hair mist\n8 fl oz (236 mL)"
    second = "JB J'ADUNÉ hydrating hair mist 8 fl oz (236 mL)"

    assert label_check.combine_reads(first, second, "J’aduné", PHOTO_REFERENCE) == ("mismatch", ["extra: (236"])


def test_brand_exact_in_one_read_is_enough_but_not_in_neither():
    exact, dropped = "JB\nJ'ADUNÉ\nBEAUTY", "JB\nJADUNÉ\nBEAUTY"

    assert label_check.combine_reads(dropped, exact, "J’aduné", "JB J’aduné BEAUTY")[0] == "unreadable"
    assert label_check.combine_reads(dropped, dropped, "J’aduné", "JB J’aduné BEAUTY") == (
        "mismatch", ["brand: expected 'J’aduné'", "extra: jaduné"]
    )


def test_reader_is_asked_to_mark_unreadable_text(vision):
    label_check.check_label(b"png", "image/png", "J’aduné")

    assert "[unreadable]" in vision.calls[0]["messages"][0]["content"][0]["text"]


# --- crop and upscale before reading ---

def png(size, color=(240, 240, 240)):
    out = io.BytesIO()
    Image.new("RGB", size, color).save(out, "PNG")
    return out.getvalue()


def sent_image(call):
    url = call["messages"][0]["content"][1]["image_url"]["url"]
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


def test_both_reads_see_the_product_cropped_and_upscaled(vision):
    image = Image.new("RGB", (1200, 1200), (240, 240, 240))
    image.paste((200, 30, 30), (400, 600, 600, 1000))  # the "bottle": columns 3, rows D-E of a 6x6 grid
    out = io.BytesIO()
    image.save(out, "PNG")
    vision.cells = ["D3", "e3"]
    vision.locate_tokens = 25_625
    vision.texts = ["JADUNÉ", "J'ADUNÉ"]

    result = label_check.check_label(out.getvalue(), "image/png", "J’aduné")

    assert result.ok and result.cropped and len(vision.calls) == 2
    [locate] = vision.locates
    assert locate["messages"][0]["content"][1]["image_url"]["detail"] == "high"
    gridded = sent_image(locate)
    assert gridded.getpixel((400, 50)) == label_check.GRID_COLOR  # the grid line between columns 2 and 3
    for call in vision.calls:
        crop = sent_image(call)
        # D3:E3 is x 400-600, y 600-1000; half a cell (100 px) each side: 400 x 600, long side to 1024
        assert crop.size == (683, 1024)
        assert crop.getpixel((341, 512)) == (200, 30, 30)  # the bottle is in the middle
        assert call["messages"][0]["content"][1]["image_url"]["detail"] == "high"
    # the locate call is paid for too: 25,625 in x $0.15/M
    assert result.cost_usd == 2 * Decimal("0.003762") + Decimal("0.00384375")


@pytest.mark.parametrize("cells, box", [
    (["A1"], (-1 / 12, -1 / 12, 1 / 6 + 1 / 12, 1 / 6 + 1 / 12)),
    (["F6", "C2", " d4 "], (1 / 6 - 1 / 12, 2 / 6 - 1 / 12, 1 + 1 / 12, 1 + 1 / 12)),
    (["Z9", "D3"], (2 / 6 - 1 / 12, 3 / 6 - 1 / 12, 3 / 6 + 1 / 12, 4 / 6 + 1 / 12)),  # invalid cells ignored
])
def test_cells_become_a_box_with_half_a_cell_of_margin(cells, box):
    got = label_check.cells_to_box(cells)

    assert (got.x0, got.y0, got.x1, got.y1) == pytest.approx(box)


@pytest.mark.parametrize("cells", [[], ["G1"], ["A7"], ["label"]])
def test_no_valid_cells_means_no_box(cells):
    assert label_check.cells_to_box(cells) is None


def test_upscaling_is_capped_at_4x():
    box = label_check.ProductBox(x0=0.4, y0=0.4, x1=0.5, y1=0.6)

    crop = Image.open(io.BytesIO(label_check.crop_product(png((500, 500)), box)))

    assert crop.size == (200, 400)  # 50 x 100 px, at most 4x


def test_large_crops_are_scaled_down_to_the_long_side():
    box = label_check.ProductBox(x0=0.1, y0=0.0, x1=0.9, y1=1.0)

    crop = Image.open(io.BytesIO(label_check.crop_product(png((2048, 2048)), box)))

    assert max(crop.size) == 1024


@pytest.mark.parametrize("box", [
    label_check.ProductBox(x0=0.5, y0=0.5, x1=0.51, y1=0.52),  # too small to be the product
    label_check.ProductBox(x0=0.5, y0=0.5, x1=0.5, y1=0.9),  # empty
])
def test_unusable_box_means_no_crop(box):
    assert label_check.crop_product(png((500, 500)), box) is None


def test_box_corners_in_either_order_and_outside_the_image_are_clamped():
    box = label_check.ProductBox(x0=1.3, y0=0.9, x1=0.6, y1=-0.2)

    assert label_check.crop_product(png((500, 500)), box) is not None


@pytest.mark.parametrize("setup", ["not_found", "locate_fails"])
def test_without_a_usable_crop_the_whole_image_is_read(vision, setup):
    image = png((600, 600))
    if setup == "locate_fails":
        vision.locate_error = RuntimeError("vision down")

    result = label_check.check_label(image, "image/png", "J’aduné")

    assert result.ok and not result.cropped and len(vision.locates) == 1
    url = vision.calls[0]["messages"][0]["content"][1]["image_url"]["url"]
    assert url == "data:image/png;base64," + base64.b64encode(image).decode()


def test_bytes_that_arent_an_image_skip_the_locate_call(vision):
    result = label_check.check_label(b"png-bytes", "image/png", "J’aduné")

    assert result.ok and not result.cropped and vision.locates == []


@pytest.mark.parametrize("read", ["JB\nJ'ADUNÉ\n[b]EAUTY", "JB\nJ'ADUNÉ\nBEA?TY"])
def test_letters_the_reader_brackets_or_marks_count_as_unreadable(read):
    assert label_check.assess(read, "J’aduné", "JB J’aduné BEAUTY")[0] == "unreadable"


def test_read_label_returns_none_instead_of_raising(vision):
    vision.error = RuntimeError("vision down")

    assert label_check.read_label(b"png", "image/png") == (None, Decimal("0"))
