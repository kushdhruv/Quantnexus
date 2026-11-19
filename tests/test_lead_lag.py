"""
Unit tests for Lead-Lag Transmission and Macro Regime Classification.
"""

import numpy as np
import pandas as pd
import pytest

from quantnexus_macro.config import MacroEventType, MacroRegime
from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager
from quantnexus_macro.analytics.lead_lag import LeadLagAnalyzer
from quantnexus_macro.analytics.regime import MacroRegimeClassifier


def test_lead_lag_analyzer():
    feed = DataFeedManager(cache_dir=None)
    lead_lag = LeadLagAnalyzer(max_lag_minutes=10)

    event = MacroRelease(
        event_id="TEST_LL",
        event_type=MacroEventType.CPI,
        name="Test CPI",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.3,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=event,
        tickers=["IEF", "SPY"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    pair_res = lead_lag.analyze_pair(windows["IEF"], windows["SPY"])
    assert pair_res.leader_ticker in ("IEF", "SPY")
    assert pair_res.optimal_lag_minutes >= 0
    assert -1.0 <= pair_res.peak_correlation <= 1.0

    velocities = lead_lag.compute_absorption_velocities(windows)
    assert "IEF" in velocities
    assert "SPY" in velocities
    assert velocities["IEF"].t50_absorption_minutes >= 0


def test_regime_classifier():
    feed = DataFeedManager(cache_dir=None)
    classifier = MacroRegimeClassifier()

    # Test Hot CPI Shock
    hot_event = MacroRelease(
        event_id="HOT_CPI",
        event_type=MacroEventType.CPI,
        name="Hot CPI",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.5,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=hot_event,
        tickers=["SPY", "TLT", "UUP", "GLD", "USO"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    res = classifier.classify_market_response(windows)
    assert res.detected_regime in (
        MacroRegime.HAWKISH_INFLATION,
        MacroRegime.GROWTH_RECESSION,
        MacroRegime.LIQUIDITY_CONTRACTION,
    )
    assert res.confidence_score > 0.40
    assert any(
        kw in res.narrative_summary
        for kw in ("Hawkish", "Discount rate", "Flight-to-Quality", "demand destruction", "contraction")
    )
