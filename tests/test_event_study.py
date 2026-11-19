"""
Unit tests for Campbell-Lo-MacKinlay Econometric Event Study Engine.
"""

import numpy as np
import pandas as pd
import pytest

from quantnexus_macro.config import MacroEventType
from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager
from quantnexus_macro.core.event_study import EventStudyEngine


def test_event_study_metrics():
    feed = DataFeedManager(cache_dir=None)
    engine = EventStudyEngine()

    event = MacroRelease(
        event_id="TEST_EVENT",
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
        tickers=["SPY", "TLT"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    metrics = engine.analyze_cross_asset_universe(windows)

    assert "SPY" in metrics
    assert "TLT" in metrics

    spy_m = metrics["SPY"]
    assert spy_m.ticker == "SPY"
    assert spy_m.t0_price > 0
    assert spy_m.final_price > 0
    assert not np.isnan(spy_m.post_event_return_pct)
    assert not np.isnan(spy_m.car_pct)
    assert not np.isnan(spy_m.t_statistic)
    assert 0.0 <= spy_m.p_value <= 1.0

    df = engine.metrics_to_dataframe(metrics)
    assert len(df) == 2
    assert "Post Return (%)" in df.columns
    assert "CAR (%)" in df.columns
