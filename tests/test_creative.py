from decimal import Decimal
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from schemas.creative import Brand
from services import creative

BRAND = Brand(brand_name="Acme", voice="playful", colors=["red"], product="sneakers", audience="runners")


def make_item(prompt):
    faq = [{"question": "Q1?", "answer": "A1."}, {"question": "Q2?", "answer": "A2."}]
    return {"prompt": prompt, "caption": "c", "alt_text": "a", "faq": faq}


def make_plan():
    return creative.CreativePlan(items=[make_item(p) for p in ("track", "beach", "city")])


@pytest.fixture
def mock_planner(monkeypatch):
    state = SimpleNamespace(calls=[])

    def create_with_completion(**kwargs):
        state.calls.append(kwargs)
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=500, total_tokens=1500)
        return make_plan(), SimpleNamespace(usage=usage)

    fake = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create_with_completion=create_with_completion)))
    monkeypatch.setattr(creative, "client", fake)
    monkeypatch.setattr(creative, "build_prompt", lambda brief, brand: f"PROMPT for {brief}")
    return state


def test_plan_creatives_returns_plan_and_token_cost(mock_planner):
    plan, cost = creative.plan_creatives("spring sale", BRAND)

    assert len(plan.items) == 3
    # 1000 * 0.15/1M + 500 * 0.60/1M
    assert cost == Decimal("0.00045")
    call = mock_planner.calls[0]
    assert call["response_model"] is creative.CreativePlan
    assert call["messages"][0]["content"] == "PROMPT for spring sale"


@pytest.mark.parametrize("items", [
    [make_item("a"), make_item("b")],                                 # too few
    [make_item("a"), make_item("b"), make_item("c"), make_item("d")],  # too many
    [make_item("a"), make_item("A "), make_item("c")],                # duplicate prompts
])
def test_plan_requires_three_distinct_items(items):
    with pytest.raises(ValidationError):
        creative.CreativePlan(items=items)


def test_item_requires_exactly_two_faqs():
    item = make_item("a")
    item["faq"] = item["faq"][:1]
    with pytest.raises(ValidationError):
        creative.CreativeItem(**item)


def test_build_prompt_is_a_stub_for_now():
    with pytest.raises(NotImplementedError):
        creative.build_prompt("brief", BRAND)
