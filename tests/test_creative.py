from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.creative import Brand, CreativeRequest
from services import creative

BRAND = Brand(brand_name="Acme", voice="playful", colors=["red"], product="sneakers", audience="runners")
REQUEST = CreativeRequest(brief="spring sale", brand=BRAND, goal="sales", platform="tiktok")


def make_item(prompt, format="ugc"):
    faq = [{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}]
    return {
        "format": format, "rationale": "Fits the brand.",
        "prompt": f"{creative.FORMAT_OPENERS.get(format, 'Photo')} of {prompt}", "caption": "c", "alt_text": "a", "faq": faq,
    }


def make_plan():
    return creative.CreativePlan(items=[
        make_item("track", "ugc"), make_item("beach", "lifestyle"), make_item("city", "flat_lay"),
    ])


@pytest.fixture
def mock_planner(monkeypatch):
    state = SimpleNamespace(calls=[], prompt_args=[])

    def create_with_completion(**kwargs):
        state.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        return make_plan(), SimpleNamespace(usage=usage)

    def fake_build_prompt(request, trend_notes):
        state.prompt_args.append((request, trend_notes))
        return f"PROMPT for {request.brief}"

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion)))
    monkeypatch.setattr(creative, "client", fake)
    monkeypatch.setattr(creative, "build_prompt", fake_build_prompt)
    return state


def test_plan_creatives_returns_plan_and_token_cost(mock_planner):
    plan, cost = creative.plan_creatives(REQUEST)

    assert len(plan.items) == 3
    # 1000 * 0.15/1M + 500 * 0.60/1M
    assert cost == Decimal("0.00045")
    call = mock_planner.calls[0]
    assert call["response_model"] is creative.CreativePlan
    assert call["messages"][0]["content"] == "PROMPT for spring sale"


def test_plan_creatives_passes_request_and_trend_notes_to_build_prompt(mock_planner):
    creative.plan_creatives(REQUEST)

    request, trend_notes = mock_planner.prompt_args[0]
    assert request is REQUEST
    assert trend_notes == creative.load_trend_notes()


@pytest.mark.parametrize("items", [
    [make_item("a", "ugc"), make_item("b", "lifestyle")],  # too few
    [make_item("a", "ugc"), make_item("b", "lifestyle"), make_item("c", "flat_lay"), make_item("d", "seasonal")],  # too many
    [make_item("a", "ugc"), make_item("b", "ugc"), make_item("c", "flat_lay")],  # duplicate formats
])
def test_plan_requires_three_distinct_items(items):
    with pytest.raises(ValidationError):
        creative.CreativePlan(items=items)


def test_item_requires_exactly_two_faqs():
    item = make_item("a")
    item["faq"] = item["faq"][:1]
    with pytest.raises(ValidationError):
        creative.CreativeItem(**item)


@pytest.mark.parametrize("format", [
    "ugc", "model_with_product", "lifestyle", "studio_hero", "flat_lay", "problem_solution", "seasonal",
])
def test_item_accepts_each_allowed_format(format):
    assert creative.CreativeItem(**make_item("a", format)).format == format


def test_item_rejects_unknown_format_and_requires_rationale():
    with pytest.raises(ValidationError):
        creative.CreativeItem(**make_item("a", "billboard"))
    item = make_item("a")
    del item["rationale"]
    with pytest.raises(ValidationError):
        creative.CreativeItem(**item)


def test_request_goal_and_platform_are_optional_and_validated():
    bare = CreativeRequest(brief="b", brand=BRAND)
    assert bare.goal is None and bare.platform is None
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, goal="retention")
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, platform="youtube")


def test_trend_notes_file_is_readable():
    assert creative.TREND_NOTES_PATH.name == "trend_notes.md"
    assert creative.load_trend_notes()


def test_missing_trend_notes_returns_empty(monkeypatch, tmp_path):
    monkeypatch.setattr(creative, "TREND_NOTES_PATH", tmp_path / "missing.md")
    assert creative.load_trend_notes() == ""





# --- brand-specific claims become placeholders ---

RETURNS = "[▢ return window, e.g. 30 days]"
SHIPPING = "[▢ shipping time, e.g. 3-5 business days]"


def test_needs_input_lists_placeholders_from_caption_and_faq_in_order():
    item = make_item("a")
    item["caption"] = f"Free returns within {RETURNS}. Ships in {SHIPPING}."
    item["faq"][1]["answer"] = f"Yes, within {RETURNS}."

    parsed = creative.CreativeItem(**item)

    assert parsed.needs_input == [RETURNS, SHIPPING]


def test_needs_input_is_rebuilt_from_the_text_not_trusted_from_the_model():
    item = make_item("a")
    item["caption"] = f"Ships in {SHIPPING}."
    item["needs_input"] = ["[▢ something not in the text]"]

    assert creative.CreativeItem(**item).needs_input == [SHIPPING]


def test_no_placeholders_means_empty_needs_input():
    assert creative.CreativeItem(**make_item("a")).needs_input == []


@pytest.mark.parametrize("field", ["prompt", "alt_text"])
def test_placeholders_rejected_in_prompt_and_alt_text(field):
    item = make_item("a")
    item[field] = f"{item[field]} {RETURNS}"

    with pytest.raises(ValidationError, match="placeholder"):
        creative.CreativeItem(**item)


def test_build_prompt_requires_claim_classes_and_placeholder_form():
    prompt = creative.build_prompt(REQUEST, "notes")

    for phrase in ("from-brief", "general-category", "brand-specific", "[▢ description, e.g. example]", "needs_input"):
        assert phrase in prompt


# --- each prompt must match its format ---

@pytest.mark.parametrize("format, opener", list(creative.FORMAT_OPENERS.items()))
def test_prompt_must_start_with_its_format_opener(format, opener):
    item = make_item("a", format)
    assert creative.CreativeItem(**item).prompt.startswith(opener)

    item["prompt"] = "Photo of trail shoes on a rock"
    with pytest.raises(ValidationError, match=f'must start with "{opener}"'):
        creative.CreativeItem(**item)


def test_opener_check_ignores_case_and_leading_space():
    item = make_item("a", "lifestyle")
    item["prompt"] = "  eye-level LIFESTYLE photo of a runner at a trailhead"
    creative.CreativeItem(**item)


@pytest.mark.parametrize("prompt_tail", [
    "trail shoes, a map and a compass arranged as a flat lay on wood",
    "shot top-down on a wooden table",
    "bird's-eye view of shoes and gear",
])
def test_flat_lay_framing_rejected_under_another_format(prompt_tail):
    # Asset 9 was labeled lifestyle but rendered as a flat lay.
    item = make_item("a", "lifestyle")
    item["prompt"] = f"Eye-level lifestyle photo, {prompt_tail}"

    with pytest.raises(ValidationError, match="describes a flat lay"):
        creative.CreativeItem(**item)


def test_flat_lay_framing_allowed_for_flat_lay_and_overhead_lighting_allowed_anywhere():
    flat = make_item("a", "flat_lay")
    flat["prompt"] = "Top-down flat lay of trail shoes, bird's-eye view, on slate"
    creative.CreativeItem(**flat)

    studio = make_item("a", "studio_hero")
    studio["prompt"] = "Studio hero shot of a trail shoe under soft overhead lighting"
    creative.CreativeItem(**studio)


def test_build_prompt_tells_the_model_each_opener():
    prompt = creative.build_prompt(REQUEST, "notes")

    for opener in creative.FORMAT_OPENERS.values():
        assert f'starts with "{opener}"' in prompt
