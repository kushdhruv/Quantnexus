"""
Unit tests for Market Data Ingestion & Jump-Diffusion Calibration.
"""

import numpy as np
import pandas as pd
import pytest

from quantnexus_macro.config import DEFAULT_ASSET_UNIVERSE, MacroEventType
from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager


def test_jump_diffusion_calibration():
    feed = DataFeedManager(cache_dir=None)

    event = MacroRelease(
        event_id="TEST_SHOCK",
        event_type=MacroEventType.CPI,
        name="Test Shock",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.3,
        consensus=3.0,
        prior=3.0,
        unit="%",
    )

    windows = feed.fetch_cross_asset_window(
        event=event,
        tickers=["SPY", "TLT", "UUP"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )

    assert "SPY" in windows
    assert "TLT" in windows
    assert "UUP" in windows

    for ticker, w in windows.items():
        assert len(w.prices) == (30 + 60 + 1)
        assert len(w.relative_minutes) == (30 + 60 + 1)
        assert w.relative_minutes[0] == -30
        assert w.relative_minutes[-1] == 60

        # Exact T0 indexing property
        t0_idx = np.where(w.relative_minutes == 0)[0][0]
        # At T0, normalized price must be exactly 100.0
        assert abs(w.normalized_prices.iloc[t0_idx] - 100.0) < 1e-4

        # Prices must be finite positive numbers
        assert np.all(w.prices.values > 0)
        assert not np.any(np.isnan(w.prices.values))

    # Test deterministic seed reproducibility
    w2 = feed.fetch_cross_asset_window(
        event=event,
        tickers=["SPY"],
        pre_event_minutes=30,
        post_event_minutes=60,
        force_simulation=True,
    )
    np.testing.assert_allclose(windows["SPY"].prices.values, w2["SPY"].prices.values)
