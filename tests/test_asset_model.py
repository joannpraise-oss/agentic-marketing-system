from decimal import Decimal

from sqlmodel import Session, select

from db.models import Asset


def make_asset(**overrides):
    fields = dict(campaign_brief="Spring sale", kind="image", prompt="a red shoe", model="m", provider="p")
    fields.update(overrides)
    return Asset(**fields)


def test_asset_defaults(engine):
    with Session(engine) as session:
        session.add(make_asset())
        session.commit()
        asset = session.exec(select(Asset)).one()

    assert asset.status == "draft"
    assert asset.cost_usd == Decimal("0")
    assert asset.file_url is None
    assert asset.format is None
    assert asset.created_at is not None


def test_asset_round_trips_faq_json_and_decimal_cost(engine):
    faq = [{"q": "Is it vegan?", "a": "Yes."}]
    with Session(engine) as session:
        session.add(make_asset(faq=faq, cost_usd=Decimal("0.0425"), status="approved", format="flat_lay"))
        session.commit()
        asset = session.exec(select(Asset)).one()

    assert asset.faq == faq
    assert asset.cost_usd == Decimal("0.0425")
    assert asset.status == "approved"
    assert asset.format == "flat_lay"
