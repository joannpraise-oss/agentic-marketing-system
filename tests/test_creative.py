from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.creative import Brand, CreativeRequest
from services import creative

BRAND = Brand(brand_name="Acme", voice="playful", colors=["red"], offer="sneakers", audience="runners")
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


@pytest.mark.parametrize("offer_type", list(creative.FORMATS_BY_OFFER_TYPE))
def test_build_prompt_lists_only_the_formats_for_the_offer_type(offer_type):
    prompt = creative.build_prompt(REQUEST.model_copy(update={"offer_type": offer_type}), "notes")

    allowed = creative.FORMATS_BY_OFFER_TYPE[offer_type]
    for fmt, opener in creative.FORMAT_OPENERS.items():
        listed = f"- {fmt}: " in prompt and f'starts with "{opener}"' in prompt
        assert listed == (fmt in allowed), fmt
    assert f"Offer type: {offer_type}" in prompt


def test_build_prompt_says_placeholders_replace_the_whole_claim():
    prompt = creative.build_prompt(REQUEST, "notes")

    assert "replaces the WHOLE unsupported claim" in prompt
    assert "[▢ return policy, e.g. free 30-day returns]" in prompt
    # The old partial-placeholder example must not come back as a model to copy.
    assert 'Wrong: "Free returns within [▢ return window, e.g. 30 days]."' in prompt
    assert prompt.count("Free returns within [▢") == 1


def test_build_prompt_forbids_hints_about_future_products():
    prompt = creative.build_prompt(REQUEST, "notes")

    assert "Do not hint at future products, releases" in prompt



# --- offer types ---

def plan_for(formats, offer_type="physical_product", count=None):
    items = [make_item(f"subject{i}", fmt) for i, fmt in enumerate(formats)]
    spec = creative.PlanSpec(count=count or len(formats), offer_type=offer_type)
    return creative.CreativePlan.model_validate({"items": items}, context={"spec": spec})


def test_every_format_has_an_opener_and_a_description():
    formats = set(creative.CreativeFormat.__args__)
    assert set(creative.FORMAT_OPENERS) == formats == set(creative.FORMAT_DESCRIPTIONS)
    for offer_type, allowed in creative.FORMATS_BY_OFFER_TYPE.items():
        assert set(allowed) <= formats
        assert len(allowed) >= creative.NUM_CREATIVES, offer_type  # 3 distinct formats must be possible


def test_request_offer_type_defaults_to_physical_product_and_is_validated():
    assert CreativeRequest(brief="b", brand=BRAND).offer_type == "physical_product"
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, offer_type="subscription")


@pytest.mark.parametrize("offer_type, formats", [
    ("physical_product", ["ugc", "studio_hero", "flat_lay"]),
    ("service", ["before_after", "experience", "problem_solution"]),
    ("content", ["mood_scene", "narrator", "lifestyle"]),
    ("idea_test", ["concept", "problem_solution", "ugc"]),
])
def test_plan_accepts_formats_allowed_for_the_offer_type(offer_type, formats):
    assert len(plan_for(formats, offer_type).items) == 3


@pytest.mark.parametrize("offer_type, bad_format", [
    ("service", "flat_lay"),          # a service has no product to lay out
    ("content", "studio_hero"),
    ("idea_test", "model_with_product"),
    ("physical_product", "narrator"),
])
def test_plan_rejects_formats_that_do_not_fit_the_offer_type(offer_type, bad_format):
    allowed = [f for f in creative.FORMATS_BY_OFFER_TYPE[offer_type] if f != bad_format][:2]
    with pytest.raises(ValidationError, match=f"{bad_format} does not fit a {offer_type} offer"):
        plan_for([*allowed, bad_format], offer_type)


def test_plan_count_comes_from_the_spec():
    with pytest.raises(ValidationError, match="exactly 3 creatives"):
        plan_for(["ugc", "studio_hero"], count=3)


def test_plan_creatives_validates_with_the_request_spec(mock_planner):
    creative.plan_creatives(REQUEST.model_copy(update={"offer_type": "service"}))

    spec = mock_planner.calls[0]["context"]["spec"]
    assert spec == creative.PlanSpec(count=3, offer_type="service")


def test_offer_type_guidance_is_a_stub_for_now():
    from tests.conftest import REAL_OFFER_TYPE_GUIDANCE

    with pytest.raises(NotImplementedError):
        REAL_OFFER_TYPE_GUIDANCE(REQUEST)
