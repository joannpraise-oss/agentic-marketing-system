import os
from decimal import Decimal

from sqlalchemy import func
from sqlmodel import Session, select

from core.errors import SpendCapError
from db.models import Asset, utcnow

DEFAULT_MAX_RUN_COST_USD = "0.25"
DEFAULT_MAX_DAILY_COST_USD = "5.00"


def max_run_cost() -> Decimal:
    return Decimal(os.getenv("MAX_RUN_COST_USD", DEFAULT_MAX_RUN_COST_USD))


def max_daily_cost() -> Decimal:
    return Decimal(os.getenv("MAX_DAILY_COST_USD", DEFAULT_MAX_DAILY_COST_USD))


def check_spend_caps(session: Session, run_cost: Decimal) -> None:
    """Raise SpendCapError before a generation is started or paid for.

    Must do:
      - If `run_cost` (this run's estimated cost so far/next step) exceeds
        max_run_cost(), raise SpendCapError("run").
      - Sum Asset.cost_usd for rows created today (UTC) and, if that sum plus
        `run_cost` exceeds max_daily_cost(), raise SpendCapError("daily").
      - Otherwise return None.

    Hint: "today UTC" = created_at >= midnight of utcnow() (db.models.utcnow is
    already naive UTC). Use select(func.coalesce(func.sum(Asset.cost_usd), 0))
    with a where on created_at, and session.exec(...).one(). Decide whether
    hitting the cap exactly is allowed (> vs >=) and keep it consistent with
    your tests.
    """
    # 1. One run too expensive?
    if run_cost > max_run_cost():
      raise SpendCapError("run")

    # 2. Today's total (UTC midnight → now) plus this run too high?
    start_of_day = utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
    spent_today = session.exec(
        select(func.coalesce(func.sum(Asset.cost_usd), 0)).where(Asset.created_at >= start_of_day)
    ).one()

    if spent_today + run_cost > max_daily_cost():
      raise SpendCapError("daily")
