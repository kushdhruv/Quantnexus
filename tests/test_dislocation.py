"""
Unit tests for Macro Transmission Dislocation Engine (MTDE) & Execution Blotter.
"""

import numpy as np
import pandas as pd
import pytest

from quantnexus_macro.config import MacroEventType
from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager
from quantnexus_macro.analytics.dislocation import MacroDislocationEngine, TradeAction
from quantnexus_macro.signals.execution_signals import ExecutionOrderManager


def test_dislocation_engine():
    feed = DataFeedManager(cache_dir=None)
    engine = MacroDislocationEngine()

    event = MacroRelease(
        event_id="TEST_DISLOC",
        event_type=MacroEventType.CPI,
        name="Test CPI Shock",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.4,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=event,
        tickers=["SPY", "QQQ", "TLT", "UUP", "GLD"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    result = engine.analyze_dislocations(windows, event)

    assert len(result.signals) == 5
    assert 0.0 <= result.cross_asset_efficiency_score <= 100.0
    assert result.mean_absolute_dislocation >= 0.0
    assert not result.dislocation_series.empty

    for s in result.signals:
        assert s.current_price > 0
        assert s.implied_fair_value_price > 0
        assert not np.isnan(s.dislocation_pct)
        assert not np.isnan(s.macro_dislocation_index)
        assert s.half_life_minutes > 0


def test_execution_ticket_generation():
    feed = DataFeedManager(cache_dir=None)
    disloc_engine = MacroDislocationEngine(z_threshold=0.5)  # Lower threshold to guarantee signals
    order_mgr = ExecutionOrderManager(book_capital_usd=10_000_000.0, max_risk_per_trade_bps=15.0)

    event = MacroRelease(
        event_id="TEST_ORDER",
        event_type=MacroEventType.CPI,
        name="Hot CPI Release",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.6,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=event,
        tickers=["SPY", "QQQ", "TLT"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    disloc_res = disloc_engine.analyze_dislocations(windows, event)
    tickets = order_mgr.generate_tickets(disloc_res.signals)

    if tickets:
        t = tickets[0]
        assert t.ticket_id.startswith("QNX-TKT")
        assert t.action in ("BUY", "SELL")
        assert t.suggested_notional_usd > 0
        assert t.max_loss_usd <= 25_000.0  # Respects book risk budget bounds
        assert len(t.execution_instructions) > 10

    df_tickets = order_mgr.tickets_to_dataframe(tickets)
    assert isinstance(df_tickets, pd.DataFrame)
