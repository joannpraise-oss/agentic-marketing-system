from decimal import Decimal

import pytest
from sqlmodel import Session

from core.errors import SpendCapError
from core.spend import check_spend_caps
from tests.test_asset_model import make_asset


def test_daily_cap_blocks_run_that_goes_over(engine):
    with Session(engine) as session:
        session.add(make_asset(cost_usd=Decimal("4.90")))   # already spent today
        session.commit()

        with pytest.raises(SpendCapError) as err:
            check_spend_caps(session, Decimal("0.20"))       # 4.90 + 0.20 = 5.10
        assert err.value.scope == "daily"


def test_hitting_daily_cap_exactly_is_allowed(engine):
    with Session(engine) as session:
        session.add(make_asset(cost_usd=Decimal("4.90")))
        session.commit()

        check_spend_caps(session, Decimal("0.10"))             # pick a cost that lands exactly on 5.00


def test_single_run_over_run_cap_is_blocked(engine):
    with Session(engine) as session:
        with pytest.raises(SpendCapError) as err:
            check_spend_caps(session, Decimal("0.30"))
        assert err.value.scope == "run"