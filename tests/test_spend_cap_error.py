from decimal import Decimal

import pytest
from fastapi import APIRouter

from core import spend
from core.errors import SpendCapError
from main import app

router = APIRouter()


@router.get("/_test/spend-cap/{scope}")
def raise_cap(scope: str):
    raise SpendCapError(scope)


app.include_router(router)


@pytest.mark.parametrize("scope", ["run", "daily"])
def test_spend_cap_maps_to_402_standard_shape(client, scope):
    resp = client.get(f"/_test/spend-cap/{scope}")

    assert resp.status_code == 402
    body = resp.json()
    assert body["code"] == "spend_cap_reached"
    assert body["scope"] == scope
    assert body["request_id"] == resp.headers["X-Request-ID"]
    assert body["detail"]


def test_caps_default_and_read_from_env(monkeypatch):
    monkeypatch.delenv("MAX_RUN_COST_USD", raising=False)
    monkeypatch.delenv("MAX_DAILY_COST_USD", raising=False)
    assert spend.max_run_cost() == Decimal("0.50")
    assert spend.max_daily_cost() == Decimal("5.00")

    monkeypatch.setenv("MAX_RUN_COST_USD", "1.5")
    monkeypatch.setenv("MAX_DAILY_COST_USD", "20")
    assert spend.max_run_cost() == Decimal("1.5")
    assert spend.max_daily_cost() == Decimal("20")
