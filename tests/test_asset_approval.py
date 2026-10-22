import pytest
from fastapi import APIRouter
from sqlmodel import Session, select

from core.placeholders import PlaceholdersRemainError
from db.models import Asset
from main import app
from tests.test_asset_model import make_asset

RETURNS = "[▢ return window, e.g. 30 days]"
SHIPPING = "[▢ shipping time, e.g. 3-5 business days]"


def draft_with_placeholders(**overrides):
    return make_asset(
        **overrides,
        caption=f"Free returns within {RETURNS}.",
        faq=[{"question": "How fast is shipping?", "answer": f"Orders arrive in {SHIPPING}."},
             {"question": "Waterproof?", "answer": "Yes."}],
    )


def test_draft_with_placeholders_can_be_saved(engine):
    with Session(engine) as session:
        session.add(draft_with_placeholders())
        session.commit()


def test_cannot_insert_approved_asset_with_placeholders(engine):
    with Session(engine) as session:
        session.add(draft_with_placeholders(status="approved"))
        with pytest.raises(PlaceholdersRemainError) as info:
            session.commit()

    assert info.value.placeholders == [RETURNS, SHIPPING]


def test_cannot_approve_until_every_placeholder_is_filled(engine):
    with Session(engine) as session:
        asset = draft_with_placeholders()
        session.add(asset)
        session.commit()

        asset.status = "approved"
        with pytest.raises(PlaceholdersRemainError) as info:
            session.commit()
        assert info.value.placeholders == [RETURNS, SHIPPING]
        session.rollback()

        # Filling only the caption still leaves the FAQ placeholder.
        asset.caption = "Free returns within 30 days."
        asset.status = "approved"
        with pytest.raises(PlaceholdersRemainError) as info:
            session.commit()
        assert info.value.placeholders == [SHIPPING]
        session.rollback()

        asset.caption = "Free returns within 30 days."
        asset.faq = [{"question": "How fast is shipping?", "answer": "Orders arrive in 3-5 business days."},
                     {"question": "Waterproof?", "answer": "Yes."}]
        asset.status = "approved"
        session.commit()

    with Session(engine) as session:
        assert session.exec(select(Asset)).one().status == "approved"


def test_archiving_with_placeholders_is_allowed(engine):
    with Session(engine) as session:
        asset = draft_with_placeholders()
        session.add(asset)
        session.commit()
        asset.status = "archived"
        session.commit()


router = APIRouter()


@router.get("/_test/approve-with-placeholders")
def approve_with_placeholders():
    raise PlaceholdersRemainError([RETURNS])


app.include_router(router)


def test_placeholders_remain_maps_to_409(client):
    resp = client.get("/_test/approve-with-placeholders")

    assert resp.status_code == 409
    body = resp.json()
    assert body["code"] == "needs_input"
    assert body["needs_input"] == [RETURNS]
    assert body["request_id"] == resp.headers["X-Request-ID"]
