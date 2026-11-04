from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.creative import FORMATS_BY_OFFER_TYPE, Brand, CreativeRequest
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
])
def test_plan_accepts_formats_allowed_for_the_offer_type(offer_type, formats):
    assert len(plan_for(formats, offer_type).items) == 3


@pytest.mark.parametrize("offer_type, bad_format", [
    ("service", "flat_lay"),          # a service has no product to lay out
    ("content", "studio_hero"),
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





# --- angles ---

def test_angles_are_optional_trimmed_distinct_and_capped():
    assert CreativeRequest(brief="b", brand=BRAND).angles is None
    assert CreativeRequest(brief="b", brand=BRAND, angles=[" moms ", "styling"]).angles == ["moms", "styling"]
    for bad in ([], ["a", "b", "c", "d", "e", "f"], ["moms", "Moms"], ["  "], ["x" * 61]):
        with pytest.raises(ValidationError):
            CreativeRequest(brief="b", brand=BRAND, angles=bad)


def test_creative_count_follows_angles():
    assert creative.creative_count(REQUEST) == 3
    assert creative.creative_count(REQUEST.model_copy(update={"angles": ["moms", "styling"]})) == 2


def angle_items(angles, formats):
    items = [make_item(f"s{i}", fmt) for i, fmt in enumerate(formats)]
    for item, angle in zip(items, angles):
        item["angle"] = angle
    return items


def test_angle_plan_allows_repeated_formats_and_normalises_angles():
    spec = creative.PlanSpec(count=2, angles=("moms", "braids and locs care"))
    items = angle_items(["Moms ", "braids and locs care"], ["lifestyle", "lifestyle"])

    plan = creative.CreativePlan.model_validate({"items": items}, context={"spec": spec})

    assert [i.angle for i in plan.items] == ["moms", "braids and locs care"]


@pytest.mark.parametrize("angles", [["styling", "moms"], ["moms", None], ["moms", "gifting"]])
def test_angle_plan_must_match_requested_angles_in_order(angles):
    spec = creative.PlanSpec(count=2, angles=("moms", "styling"))
    with pytest.raises(ValidationError, match="one creative per angle"):
        creative.CreativePlan.model_validate(
            {"items": angle_items(angles, ["ugc", "lifestyle"])}, context={"spec": spec}
        )


@pytest.mark.parametrize("returned", [None, "", "Peptide care for soft, hydrated hair "])
def test_single_angle_plan_fills_in_an_omitted_angle(returned):
    # The live failure: one angle requested, the model left `angle` out.
    spec = creative.PlanSpec(count=1, angles=("peptide care for soft, hydrated hair",))

    plan = creative.CreativePlan.model_validate(
        {"items": angle_items([returned], ["model_with_product"])}, context={"spec": spec}
    )

    assert [i.angle for i in plan.items] == ["peptide care for soft, hydrated hair"]


def test_single_angle_prompt_asks_for_one_concept_not_one_per_format():
    # The live failure: "exactly 1 ad concepts ... formats may repeat" made the model return 5-7 items.
    prompt = creative.build_prompt(REQUEST.model_copy(update={"angles": ["peptide care"]}), "")

    assert "Create exactly ONE ad concept (items holds a single item) for the angle below." in prompt
    assert "Do not make a concept for each format." in prompt
    assert "ANGLE: peptide care" in prompt
    assert "1 ad concepts" not in prompt and "formats may repeat" not in prompt


def test_single_creative_plan_with_several_items_is_re_asked_for_one():
    spec = creative.PlanSpec(count=1, angles=("moms",))
    items = angle_items(["moms", "moms"], ["ugc", "lifestyle"])
    with pytest.raises(ValidationError, match="exactly 1 creative in items, not 2: one concept, not one per format"):
        creative.CreativePlan.model_validate({"items": items}, context={"spec": spec})


def test_single_angle_plan_still_rejects_a_different_angle():
    spec = creative.PlanSpec(count=1, angles=("moms",))
    with pytest.raises(ValidationError, match="one creative per angle"):
        creative.CreativePlan.model_validate({"items": angle_items(["gifting"], ["ugc"])}, context={"spec": spec})


def test_plan_without_angles_drops_any_angle_the_model_invents():
    plan = creative.CreativePlan.model_validate(
        {"items": angle_items(["x", "y", "z"], ["ugc", "lifestyle", "flat_lay"])}
    )
    assert [i.angle for i in plan.items] == [None, None, None]


def test_build_prompt_lists_angles_in_order():
    request = REQUEST.model_copy(update={"angles": ["moms", "styling"]})

    prompt = creative.build_prompt(request, "")

    assert "Create exactly 2 ad concepts for the same offer" in prompt
    assert "1. moms\n2. styling" in prompt
    assert "formats may repeat" in prompt
    assert "Create exactly 3" in creative.build_prompt(REQUEST, "")


def test_plan_creatives_spec_carries_angles(mock_planner):
    creative.plan_creatives(REQUEST.model_copy(update={"angles": ["moms", "styling"]}))

    spec = mock_planner.calls[0]["context"]["spec"]
    assert spec.count == 2 and spec.angles == ("moms", "styling")



# --- source text rights and placeholder placement ---

def test_source_text_requires_rights_flag():
    with pytest.raises(ValidationError, match="has_text_rights"):
        CreativeRequest(brief="b", brand=BRAND, source_text="Chapter one began at dawn.")
    ok = CreativeRequest(brief="b", brand=BRAND, source_text="Chapter one began at dawn.", has_text_rights=True)
    assert ok.source_text == "Chapter one began at dawn."


def test_blank_source_text_is_ignored_and_length_is_capped():
    assert CreativeRequest(brief="b", brand=BRAND, source_text="   ").source_text is None
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, source_text="x" * 2001, has_text_rights=True)


def test_build_prompt_includes_source_text_only_when_given():
    request = REQUEST.model_copy(update={"source_text": "Dawn came slowly.", "has_text_rights": True})

    assert "SOURCE TEXT" in creative.build_prompt(request, "")
    assert "Dawn came slowly." in creative.build_prompt(request, "")
    assert "SOURCE TEXT (" not in creative.build_prompt(REQUEST, "")


def test_build_prompt_keeps_brand_facts_in_faq():
    prompt = creative.build_prompt(REQUEST, "")

    assert "Keep brand-specific facts in the FAQ" in prompt
    assert "Quote only from SOURCE TEXT" in prompt


# --- on-image text ---

def text_request(**fields):
    return CreativeRequest(brief="b", brand=BRAND, **fields)


def test_text_defaults_to_none_and_overlay():
    request = text_request()
    assert (request.text_choice, request.text_mode, request.on_image_text) == ("none", "overlay", None)


def test_own_text_one_line_for_all_or_one_per_creative():
    assert text_request(text_choice="own", on_image_text=" Run further ").own_texts() == ["Run further"] * 3
    per = text_request(text_choice="own", on_image_text=["A", "B"], angles=["moms", "styling"])
    assert per.own_texts() == ["A", "B"]


@pytest.mark.parametrize("fields, message", [
    ({"text_choice": "own"}, "needs on_image_text"),
    ({"text_choice": "own", "on_image_text": ["A", "B"]}, "2 lines but the run makes 3"),
    ({"text_choice": "own", "on_image_text": " ".join(["word"] * 13)}, "at most 12 words"),
    ({"text_choice": "own", "on_image_text": "one two three four five six", "text_mode": "rendered"}, "at most 5 words"),
    ({"text_choice": "own", "on_image_text": "Free returns [▢ policy, e.g. 30 days]"}, "placeholders"),
    ({"text_choice": "suggest", "on_image_text": "Mine"}, "only sent with text_choice 'own'"),
    ({"on_image_text": "Mine"}, "only sent with text_choice 'own'"),
])
def test_on_image_text_request_rules(fields, message):
    with pytest.raises(ValidationError, match=message):
        text_request(**fields)


def text_plan(items_fields, **spec_fields):
    formats = ["ugc", "lifestyle", "flat_lay"]
    items = [make_item(f"s{i}", fmt) for i, fmt in enumerate(formats)]
    for item, fields in zip(items, items_fields):
        item.update(fields)
    spec = creative.PlanSpec(**spec_fields)
    return creative.CreativePlan.model_validate({"items": items}, context={"spec": spec})


OVERLAY = "Candid smartphone photo of a runner, clean negative space in the top third"


def test_none_drops_any_text_the_model_adds():
    plan = text_plan([{"on_image_text": "Buy now"}] * 3)
    assert [i.on_image_text for i in plan.items] == [None] * 3


def test_own_text_overrides_the_model_and_overlay_needs_negative_space():
    fields = [{"prompt": OVERLAY, "on_image_text": "model words"},
              {"prompt": "Eye-level lifestyle photo, negative space on the left"},
              {"prompt": "Top-down flat lay, clean negative space at the bottom"}]
    plan = text_plan(fields, text_choice="own", own_texts=("Run further",) * 3)

    assert [i.on_image_text for i in plan.items] == ["Run further"] * 3


def test_overlay_prompt_without_negative_space_is_rejected():
    with pytest.raises(ValidationError, match="negative space"):
        text_plan([], text_choice="own", own_texts=("Run",) * 3)


def test_overlay_text_must_stay_out_of_the_image_prompt():
    fields = [{"prompt": f'{OVERLAY}, with the words "Run further"'}]
    with pytest.raises(ValidationError, match="keep \"Run further\" out of the image prompt"):
        text_plan(fields, text_choice="own", own_texts=("Run further",) * 3)


def test_overlay_allows_the_word_inside_normal_prose():
    # "calm" as text and "calm morning light" in the scene is fine; only the quoted text is banned.
    fields = [{"prompt": "Candid smartphone photo, calm morning light, negative space on top"},
              {"prompt": "Eye-level lifestyle photo, negative space on the left"},
              {"prompt": "Top-down flat lay, negative space at the bottom"}]
    text_plan(fields, text_choice="own", own_texts=("Calm",) * 3)


def test_rendered_prompt_must_contain_the_exact_quoted_text():
    openers = ["Candid smartphone photo", "Eye-level lifestyle photo", "Top-down flat lay"]
    good = [{"prompt": f'{opener}, bold white text "Run {word}" across the top'}
            for opener, word in zip(openers, ("far", "fast", "free"))]
    plan = text_plan(good, text_choice="own", text_mode="rendered", own_texts=("Run far", "Run fast", "Run free"))
    assert [i.on_image_text for i in plan.items] == ["Run far", "Run fast", "Run free"]

    with pytest.raises(ValidationError, match='must contain the exact text "Run far"'):
        text_plan([{"prompt": "Candid smartphone photo, text RUN FAR on top"}],
                  text_choice="own", text_mode="rendered", own_texts=("Run far", "Run fast", "Run free"))


@pytest.mark.parametrize("text, message", [
    (None, "needs on_image_text"),
    (" ".join(["w"] * 13), "at most 12 words"),
    ("Returns: [▢ policy, e.g. free 30-day returns]", "placeholders"),
])
def test_suggested_text_is_required_short_and_placeholder_free(text, message):
    fields = [{"prompt": OVERLAY, "on_image_text": text}]
    with pytest.raises(ValidationError, match=message):
        text_plan(fields, text_choice="suggest")


def test_suggested_text_accepted():
    fields = [{"prompt": OVERLAY, "on_image_text": "Trails don't wait"},
              {"prompt": "Eye-level lifestyle photo, negative space on the left", "on_image_text": "Grip that holds"},
              {"prompt": "Top-down flat lay, negative space at the bottom", "on_image_text": "Pack light"}]
    plan = text_plan(fields, text_choice="suggest")
    assert plan.items[0].on_image_text == "Trails don't wait"


@pytest.mark.parametrize("fields, expected", [
    ({}, "ON-IMAGE TEXT: none"),
    ({"text_choice": "suggest"}, "Mode: OVERLAY"),
    ({"text_choice": "suggest", "text_mode": "rendered"}, "Mode: RENDERED"),
    ({"text_choice": "own", "on_image_text": "Run further"}, '1. "Run further"'),
])
def test_build_prompt_text_section(fields, expected):
    prompt = creative.build_prompt(REQUEST.model_copy(update=fields), "")
    assert expected in prompt
    assert "No words in the image, except the exact" in prompt
    assert "No placeholders in on_image_text" in prompt


def test_spec_from_request_carries_text_settings():
    request = text_request(text_choice="own", on_image_text="Go", text_mode="rendered")
    spec = creative.PlanSpec.from_request(request)
    assert (spec.text_choice, spec.text_mode, spec.own_texts) == ("own", "rendered", ("Go", "Go", "Go"))



# --- strategy notes ---

def test_strategy_notes_optional_trimmed_and_capped():
    assert CreativeRequest(brief="b", brand=BRAND).strategy_notes is None
    assert CreativeRequest(brief="b", brand=BRAND, strategy_notes="   ").strategy_notes is None
    assert CreativeRequest(brief="b", brand=BRAND, strategy_notes=" Lead with comfort. ").strategy_notes == "Lead with comfort."
    CreativeRequest(brief="b", brand=BRAND, strategy_notes="x" * 1500)
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, strategy_notes="x" * 1501)


def test_build_prompt_shows_strategy_notes_after_trend_notes():
    request = REQUEST.model_copy(update={"strategy_notes": "Lead with comfort, not speed."})

    prompt = creative.build_prompt(request, "UGC beats studio on TikTok.")

    assert "STRATEGY NOTES" in prompt and "Lead with comfort, not speed." in prompt
    assert prompt.index("TREND NOTES") < prompt.index("STRATEGY NOTES") < prompt.index("RULES:")


def test_build_prompt_omits_strategy_section_without_notes():
    assert "STRATEGY NOTES" not in creative.build_prompt(REQUEST, "notes")



def test_idea_test_offer_type_and_concept_format_are_gone():
    with pytest.raises(ValidationError):
        CreativeRequest(brief="b", brand=BRAND, offer_type="idea_test")
    assert "concept" not in creative.FORMAT_OPENERS
    assert set(creative.FORMATS_BY_OFFER_TYPE) == {"physical_product", "service", "content"}


# --- creative tests ---

def ab_request(**fields):
    return CreativeRequest(brief="b", brand=BRAND, **fields)


def test_creative_test_defaults_off():
    request = ab_request()
    assert (request.creative_test, request.test_format, request.vary_format) == (False, None, False)


@pytest.mark.parametrize("fields, message", [
    ({"creative_test": True}, "needs 2-5 angles"),
    ({"creative_test": True, "angles": ["moms"]}, "needs 2-5 angles"),
    ({"creative_test": True, "angles": ["a", "b", "c", "d", "e", "f"]}, "at most 5"),
    ({"test_format": "ugc"}, "only used with creative_test"),
    ({"vary_format": True}, "only used with creative_test"),
    ({"creative_test": True, "angles": ["a", "b"], "test_format": "ugc", "vary_format": True}, "not both"),
    ({"creative_test": True, "angles": ["a", "b"], "test_format": "narrator"}, "does not fit a physical_product"),
    ({"creative_test": True, "angles": ["a", "b"], "test_format": "billboard"}, "test_format"),
])
def test_creative_test_request_rules(fields, message):
    with pytest.raises(ValidationError, match=message):
        ab_request(**fields)


COMPOSITION = "on a sunlit trail at golden hour, mid-shot, natural colors"


def ab_items(formats, angles=("moms", "styling"), with_composition=True, core=True):
    items = []
    for i, (fmt, angle) in enumerate(zip(formats, angles)):
        item = make_item(f"person{i}" + (f", {COMPOSITION}" if with_composition else ""), fmt)
        item["angle"] = angle
        if core:
            item["core_message"] = f"Core message for {angle}"
        items.append(item)
    return items


def validate_test(items, plan_fields=None, **spec_fields):
    spec = creative.PlanSpec(count=len(items), angles=tuple(i["angle"] for i in items), creative_test=True, **spec_fields)
    return creative.CreativePlan.model_validate({"items": items, **(plan_fields or {})}, context={"spec": spec})


RATIONALE = {"format_rationale": "UGC reads as native on TikTok for every angle.", "shared_composition": COMPOSITION}


def test_single_format_test_with_chosen_format_needs_rationale_and_composition():
    plan = validate_test(ab_items(["ugc", "ugc"]), RATIONALE)

    assert plan.format_rationale.startswith("UGC reads")
    assert plan.shared_composition == COMPOSITION
    assert [i.core_message for i in plan.items] == ["Core message for moms", "Core message for styling"]


@pytest.mark.parametrize("plan_fields, message", [
    ({"shared_composition": COMPOSITION}, "format_rationale"),
    ({"format_rationale": "Because."}, "shared_composition"),
])
def test_single_format_test_missing_plan_fields(plan_fields, message):
    with pytest.raises(ValidationError, match=message):
        validate_test(ab_items(["ugc", "ugc"]), plan_fields)


def test_single_format_test_rejects_mixed_formats():
    with pytest.raises(ValidationError, match="same format"):
        validate_test(ab_items(["ugc", "lifestyle"]), RATIONALE)


def test_test_format_is_enforced_and_needs_no_rationale():
    plan = validate_test(ab_items(["studio_hero", "studio_hero"]), {"shared_composition": COMPOSITION},
                         test_format="studio_hero")
    assert plan.format_rationale is None

    with pytest.raises(ValidationError, match="must use the format studio_hero"):
        validate_test(ab_items(["ugc", "ugc"]), RATIONALE, test_format="studio_hero")


def test_every_prompt_must_repeat_the_shared_composition():
    items = ab_items(["ugc", "ugc"])
    items[1]["prompt"] = "Candid smartphone photo of person1 in a city cafe"

    with pytest.raises(ValidationError, match="Creative 2: the image prompt must include shared_composition"):
        validate_test(items, RATIONALE)


@pytest.mark.parametrize("core, message", [
    (None, "core_message must be one line"),
    ("line one\nline two", "core_message must be one line"),
    (" ".join(["w"] * 21), "at most 20 words"),
    ("Free returns: [▢ return policy, e.g. free 30-day returns]", "placeholders"),
])
def test_core_message_is_one_short_line(core, message):
    items = ab_items(["ugc", "ugc"])
    items[0]["core_message"] = core
    with pytest.raises(ValidationError, match=message):
        validate_test(items, RATIONALE)


def test_vary_format_allows_different_formats_without_shared_composition():
    plan = validate_test(ab_items(["ugc", "lifestyle"], with_composition=False), RATIONALE, vary_format=True)

    assert [i.format for i in plan.items] == ["ugc", "lifestyle"]
    assert plan.format_rationale is None and plan.shared_composition is None


def test_outside_a_test_the_test_fields_are_dropped():
    items = ab_items(["ugc", "lifestyle"])
    spec = creative.PlanSpec(count=2, angles=("moms", "styling"))

    plan = creative.CreativePlan.model_validate({"items": items, **RATIONALE}, context={"spec": spec})

    assert plan.format_rationale is None and plan.shared_composition is None
    assert [i.core_message for i in plan.items] == [None, None]


def test_build_prompt_creative_test_section():
    request = ab_request(creative_test=True, angles=["moms", "styling"])

    prompt = creative.build_prompt(request, "")

    assert "CREATIVE TEST" in prompt
    assert "differ ONLY in its\nmessaging angle - the hook, the person shown and the caption" in prompt
    assert "Pick ONE format" in prompt and "format_rationale" in prompt
    assert "shared_composition" in prompt
    assert "core message in one line" in prompt
    assert "1. moms\n2. styling" in prompt


def test_build_prompt_creative_test_with_test_format_and_vary_format():
    fixed = creative.build_prompt(ab_request(creative_test=True, angles=["a", "b"], test_format="ugc"), "")
    assert "Use the format ugc for EVERY concept" in fixed and "Pick ONE format" not in fixed

    varied = creative.build_prompt(ab_request(creative_test=True, angles=["a", "b"], vary_format=True), "")
    assert "formats may differ" in varied and "shared_composition" not in varied


def test_plain_angles_prompt_is_not_a_creative_test():
    prompt = creative.build_prompt(ab_request(angles=["a", "b"]), "")
    assert "CREATIVE TEST" not in prompt
    assert "core_message: leave empty" in prompt


def test_spec_carries_creative_test_settings():
    spec = creative.PlanSpec.from_request(ab_request(creative_test=True, angles=["a", "b"], test_format="ugc"))
    assert (spec.creative_test, spec.test_format, spec.vary_format, spec.single_format_test) == (True, "ugc", False, True)


PHOTO_REQUEST = REQUEST.model_copy(update={"product_photo_path": "f51db36f3c08442daae2a6325a563aac.jpg"})


def test_product_label_uses_the_wording_that_passed_the_visual_test():
    assert creative.with_product_label("Top-down photo of the bottle.", "J’aduné", "flat_lay") == (
        "Top-down photo of the bottle. The product label reads exactly «J’aduné» with the apostrophe. "
        "Keep all label text exactly as in the reference photo; do not change or invent any text. "
        "Do not add objects beyond those described in this prompt."
    )


@pytest.mark.parametrize("format", ["studio_hero", "model_with_product"])
def test_label_formats_require_a_legible_front_facing_label_at_true_size(format):
    prompt = creative.with_product_label("Photo of a woman holding the bottle.", "J’aduné", format)

    assert prompt.startswith(
        "Photo of a woman holding the bottle. The product faces the camera; the label is in focus and legible, "
        "and the product stays at true real-world size, even if that makes the label smaller. "
        "The product label reads exactly «J’aduné»"
    )
    assert "25%" not in prompt


@pytest.mark.parametrize("format", ["ugc", "lifestyle", "flat_lay", "problem_solution", "seasonal"])
def test_other_formats_keep_their_own_framing(format):
    assert creative.LABEL_FACING_RULE not in creative.with_product_label("Photo of the bottle.", "J’aduné", format)


@pytest.mark.parametrize("format, prompt", [
    ("ugc", "Candid smartphone photo of a woman misting her hair."),
    ("lifestyle", "Eye-level lifestyle photo of a woman reading by the window."),
])
def test_in_use_formats_let_the_hand_cover_the_label_and_keep_true_size(format, prompt):
    prompt = finalize(prompt, format, has_photo=True, product_size=SIZE)

    assert (
        "The product is handled naturally while in use: the hand may cover part of the label or the product "
        "may be turned, and it stays at true real-world size."
    ) in prompt
    assert "faces the camera" not in prompt and creative.HOLD_LOWER_HALF not in prompt
    assert f"The product ({SIZE}) is shown at true real-world size" in prompt
    assert "The product label reads exactly «J’aduné»" in prompt  # visible text must still be exact


def test_lifestyle_without_people_may_turn_the_product_but_adds_no_hand():
    prompt = finalize("Eye-level lifestyle photo of the bottle on a vanity with candles.", "lifestyle", has_photo=True)

    assert creative.IN_USE_LABEL_RULE_NO_PEOPLE in prompt and "hand" not in prompt
    assert "faces the camera" not in prompt


def test_planner_is_told_people_formats_need_a_readable_label():
    prompt = creative.build_prompt(PHOTO_REQUEST, "")

    assert "In studio_hero and model_with_product, the product faces the camera" in prompt
    assert "In ugc and lifestyle, show the product\nnaturally in use" in prompt
    assert "True size wins over label size" in prompt and "25%" not in prompt


def test_props_rule_bans_lookalike_containers_and_encourages_mood_props():
    prompt = creative.build_prompt(REQUEST, "")

    assert "Never include unbranded product containers" in prompt
    assert "mistaken for other products from this brand" in prompt
    assert "Mood props like candles, flowers, mirrors and fabrics are encouraged for lifestyle and\n  luxury formats." in prompt


@pytest.mark.parametrize("offer_type", list(creative.FORMATS_BY_OFFER_TYPE))
def test_realism_rule_is_in_every_plan(offer_type):
    # Every offer type allows at least one format with people in it.
    prompt = creative.build_prompt(REQUEST.model_copy(update={"offer_type": offer_type}), "")

    assert creative.REALISM_RULE in prompt
    flat = " ".join(prompt.split())
    for phrase in [
        "ugc, model_with_product, lifestyle with people, before_after, experience, narrator",
        "People must look like real, photographed humans, not AI-perfect.",
        "visible pores and fine texture",
        "freckles, moles, small spots, faint veins, or piercings",
        "Keep skin healthy and glowing, with real depth",
        "fully clothed in everyday wear suited to the brand (e.g. a linen shirt or knit top)",
        "framed head-and-shoulders or waist-up",
        "varied skin color within the face",
        "Avoid airbrushed or plastic skin, perfect symmetry, and over-smoothed hair.",
        "don't give everyone the same features",
        "natural falloff, soft shadows, slight grain where it fits UGC.",
    ]:
        assert phrase in flat


def test_planner_is_told_about_the_reference_photo_only_when_there_is_one():
    with_photo = creative.build_prompt(PHOTO_REQUEST, "")

    assert "PRODUCT PHOTO:" in with_photo
    assert "describes the scene" in with_photo and "do not redesign it" in with_photo
    assert "PRODUCT PHOTO:" not in creative.build_prompt(REQUEST, "")


@pytest.mark.parametrize("request_", [REQUEST, PHOTO_REQUEST])
def test_physical_product_prompts_get_the_props_rule(request_):
    prompt = creative.build_prompt(request_, "")

    assert "Props are allowed only when you deliberately choose them" in prompt
    assert "Never add props that imply ingredients,\n  results or features not stated in the brief." in prompt
    assert "The product stays the focal point." in prompt
    assert "No added props" not in prompt


def test_props_rule_is_left_out_when_there_is_no_product():
    prompt = creative.build_prompt(REQUEST.model_copy(update={"offer_type": "content"}), "")

    assert "Props are allowed only" not in prompt


# --- rules appended to every image prompt ---

def finalize(prompt, format, offer_type="physical_product", has_photo=False, product_size=None, safer=False):
    return creative.finalize_image_prompt(prompt, format, offer_type, "J’aduné", has_photo, product_size, safer)


SIZE = "8 fl oz spray bottle, about 7 in tall"


def test_product_size_sets_true_scale_against_hands_and_face():
    prompt = finalize("Candid smartphone photo of a woman misting her hair.", "ugc", product_size=SIZE)

    assert (
        "The product (8 fl oz spray bottle, about 7 in tall) is shown at true real-world size relative to "
        "the hands and face, no larger than a hand from wrist to fingertip."
    ) in prompt
    assert creative.SCALE_DEFAULT not in prompt


def test_product_only_shots_scale_against_surroundings_not_hands():
    prompt = finalize("Photo of the bottle on linen.", "studio_hero", product_size=SIZE)

    assert f"The product ({SIZE}) is shown at true real-world size relative to its surroundings." in prompt
    assert "hand" not in prompt  # mentioning a hand would invite one into the shot


@pytest.mark.parametrize("format", ["ugc", "studio_hero"])
def test_without_product_size_the_product_is_not_enlarged(format):
    prompt = finalize("Photo of the scene.", format)

    assert "The product is shown at realistic real-world size, not enlarged." in prompt
    assert "true real-world size" not in prompt


def test_held_product_is_separated_from_hand_and_held_by_the_lower_half():
    prompt = finalize("Photo of a fictional model holding the bottle.", "model_with_product")

    assert (
        "The product is in sharp focus and clearly separated from the hand and background by contrast in tone "
        "or lighting, with a soft highlight along its edge."
    ) in prompt
    assert "The hand holds the lower half so the label stays fully visible." in prompt


@pytest.mark.parametrize("format", sorted(set(FORMATS_BY_OFFER_TYPE["physical_product"]) - {"ugc", "model_with_product"}))
def test_every_product_format_gets_scale_and_separation(format):
    prompt = finalize("Photo of the bottle on linen.", format)

    assert creative.SCALE_DEFAULT in prompt
    assert "is in sharp focus and clearly separated from the background" in prompt
    assert creative.HOLD_LOWER_HALF not in prompt


def test_other_offer_types_get_no_scale_or_separation():
    prompt = finalize("Photo of a fictional narrator.", "narrator", "content", product_size=SIZE)

    assert "real-world size" not in prompt and "sharp focus" not in prompt


def test_product_rules_come_before_the_label_rules():
    prompt = finalize("Photo of a fictional model holding the bottle.", "model_with_product", has_photo=True,
                      product_size=SIZE)

    one = prompt.index(creative.ONE_PRODUCT_HELD)
    scale = prompt.index("true real-world size")
    hold = prompt.index(creative.HOLD_LOWER_HALF)
    facing = prompt.index(creative.LABEL_FACING_RULE)
    assert one < scale < hold < facing


@pytest.mark.parametrize("format", ["ugc", "model_with_product", "before_after", "experience", "narrator"])
def test_people_formats_get_the_realism_rule(format):
    assert creative.REALISM_PROMPT in finalize("Photo of the scene.", format, "service")


def test_realism_prompt_is_the_wording_of_the_golden_runs():
    # #25 and #26 (docs/golden/), made at bcecaa5.
    assert creative.REALISM_PROMPT == (
        "Real photographed human, not AI-perfect: visible pores and fine skin texture, natural unevenness, "
        "freckles/moles/spots where natural, healthy glow with real shadows and undertones, no airbrushing, "
        "natural camera grain."
    )


def test_no_prompt_wording_says_editorial():
    # "editorial" implies retouching: the live images came back poreless and glossy.
    for wording in (creative.REALISM_PROMPT, creative.SAFER_REALISM_PROMPT):
        assert "editorial" not in wording.lower()


@pytest.mark.parametrize("format", ["ugc", "model_with_product", "before_after", "experience", "narrator"])
def test_people_formats_state_wardrobe_and_framing(format):
    assert finalize("Photo of the scene.", format, "service").endswith(
        " The person is fully clothed in everyday wear suited to the brand (e.g. a linen shirt or knit top), "
        "framed head-and-shoulders or waist-up."
    )


def test_product_only_formats_get_no_wardrobe_rule():
    assert creative.WARDROBE_FRAMING not in finalize("Photo of the bottle on linen.", "studio_hero")


def test_people_prompts_keep_the_planners_skin_wording():
    # Only wardrobe and framing were added for the content filter; the scene is passed on as written.
    planner = "Portrait of a woman with glowing skin and faint veins on her temple, holding the bottle."

    assert finalize(planner, "model_with_product").startswith(planner)


@pytest.mark.parametrize("planner, expected", [
    ("Photo of a woman in a white robe, holding the bottle.", "Photo of a woman in a white linen shirt, holding the bottle."),
    ("She wears a simple, elegant white bathrobe.", "She wears a simple, elegant white linen shirt."),
    ("A woman wrapped in a fluffy towel holds the bottle.", "A woman wearing a linen shirt holds the bottle."),
    ("Casual, elegant attire, like a light linen robe.", "Casual, elegant attire, like a light linen shirt."),
])
def test_people_prompts_swap_robes_and_worn_towels_for_everyday_clothing(planner, expected):
    assert finalize(planner, "model_with_product").startswith(expected)


def test_towels_as_decor_stay():
    planner = "Photo of a woman holding the bottle; soft towels hang on the rack behind her."

    assert finalize(planner, "ugc").startswith(planner)


def test_product_only_prompts_keep_their_robes_and_towels():
    planner = "Photo of the bottle on a folded towel beside a robe hook."

    assert finalize(planner, "studio_hero").startswith(planner)


def test_safer_retry_wording_has_no_skin_detail_and_swaps_risky_scene_words():
    planner = "Sensual photo of a woman in a tank top in her bathroom, bare shoulders, dewy skin, holding the bottle."

    prompt = finalize(planner, "model_with_product", has_photo=True, safer=True)

    lowered = prompt.lower()
    for word in ("sensual", "tank top", "bathroom", "bare", "skin", "pores", "freckles"):
        assert word not in lowered, word
    assert creative.SAFER_REALISM_PROMPT in prompt and creative.WARDROBE_FRAMING in prompt
    assert creative.REALISM_PROMPT not in prompt
    assert "The product label reads exactly «J’aduné»" in prompt  # the product and label rules stay


def test_safer_wording_leaves_product_only_prompts_alone():
    planner = "Photo of the bottle on a bathroom shelf."

    assert finalize(planner, "studio_hero", safer=True) == finalize(planner, "studio_hero")


@pytest.mark.parametrize("prompt, people", [
    ("Eye-level lifestyle photo of a woman reading by the window.", True),
    ("Eye-level lifestyle photo of friends at a picnic.", True),
    ("Eye-level lifestyle photo of the bottle on a vanity with candles.", False),
])
def test_lifestyle_gets_the_realism_rule_only_when_it_shows_people(prompt, people):
    assert (creative.REALISM_PROMPT in finalize(prompt, "lifestyle")) is people


@pytest.mark.parametrize("format", ["studio_hero", "flat_lay"])
def test_product_only_formats_get_no_realism_rule(format):
    assert creative.REALISM_PROMPT not in finalize("Photo of the bottle.", format)


def test_people_with_a_product_get_one_held_product():
    prompt = finalize("Candid smartphone photo of a woman misting her hair.", "ugc")

    assert "Exactly one product in the image; the person holds and uses the same single product." in prompt


def test_product_without_people_gets_exactly_one_product():
    prompt = finalize("Photo of the bottle on linen.", "studio_hero")

    assert " Exactly one product in the image. " in prompt and creative.ONE_PRODUCT_HELD not in prompt


def test_other_offer_types_get_no_product_rule():
    prompt = finalize("Photo of a fictional narrator.", "narrator", "content")

    assert creative.ONE_PRODUCT not in prompt and creative.REALISM_PROMPT in prompt


def test_photo_runs_end_with_the_label_rules_after_realism_and_one_product():
    prompt = finalize("Photo of a fictional model holding the bottle.", "model_with_product", has_photo=True)

    realism = prompt.index(creative.REALISM_PROMPT)
    one = prompt.index(creative.ONE_PRODUCT_HELD)
    facing = prompt.index(creative.LABEL_FACING_RULE)
    label = prompt.index("The product label reads exactly «J’aduné»")
    assert realism < one < facing < label
    assert prompt.endswith("Do not add objects beyond those described in this prompt.")


def test_runs_without_a_photo_get_no_label_rules():
    prompt = finalize("Candid smartphone photo of a woman misting her hair.", "ugc")

    assert "The product label reads" not in prompt and "reference photo" not in prompt
