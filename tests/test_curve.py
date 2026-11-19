"""
Unit tests for Fixed Income Yield Curve & Curve Spread Analytics.
"""

import numpy as np
import pandas as pd
import pytest

from quantnexus_macro.config import MacroEventType
from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager
from quantnexus_macro.analytics.curve import YieldCurveAnalyzer, CurveRegime


def test_yield_curve_shifts_and_regimes():
    feed = DataFeedManager(cache_dir=None)
    analyzer = YieldCurveAnalyzer()

    # Hawkish shock: CPI beat
    event = MacroRelease(
        event_id="TEST_HAWKISH",
        event_type=MacroEventType.CPI,
        name="Test Hot CPI",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.4,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=event,
        tickers=["SHY", "IEI", "IEF", "TLT"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    res = analyzer.analyze_curve_response(windows)
    assert res is not None

    # Tenors 2, 5, 10, 20 must exist
    assert 2 in res.tenor_shifts
    assert 5 in res.tenor_shifts
    assert 10 in res.tenor_shifts
    assert 20 in res.tenor_shifts

    # Under hawkish shock, yields should rise (positive yield change bps)
    assert res.tenor_shifts[2].yield_change_bps > 0
    assert res.tenor_shifts[10].yield_change_bps > 0

    # Intraday slope series must match window length
    assert len(res.intraday_slope_series) == (30 + 60 + 1)

    # Test direct classification logic
    regime_bf, _ = analyzer._classify_curve_regime(dy2=15.0, dy10=8.0, slope_delta=-7.0)
    assert regime_bf == CurveRegime.BEAR_FLATTENER

    regime_bs, _ = analyzer._classify_curve_regime(dy2=5.0, dy10=18.0, slope_delta=13.0)
    assert regime_bs == CurveRegime.BEAR_STEEPENER

    regime_bull_s, _ = analyzer._classify_curve_regime(dy2=-20.0, dy10=-5.0, slope_delta=15.0)
    assert regime_bull_s == CurveRegime.BULL_STEEPENER
