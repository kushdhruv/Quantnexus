"""
Unit tests for Economic Calendar and Standardized Macro Surprise Engine.
"""

import pandas as pd
import pytest

from quantnexus_macro.config import MacroEventType
from quantnexus_macro.core.calendar import EconomicCalendar, MacroRelease


def test_calendar_surprise_and_z_score():
    rel = MacroRelease(
        event_id="TEST_CPI",
        event_type=MacroEventType.CPI,
        name="Test CPI Release",
        timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
        actual=3.1,
        consensus=2.9,
        prior=3.4,
        unit="%",
    )

    # Surprise = 3.1 - 2.9 = 0.2%
    assert abs(rel.surprise - 0.20) < 1e-6
    assert abs(rel.surprise_prior - (-0.30)) < 1e-6

    # For CPI, historical sigma is 0.15
    # Z = 0.20 / 0.15 = 1.333...
    assert abs(rel.z_score - (0.20 / 0.15)) < 1e-4
    assert rel.is_significant is True
    assert rel.is_extreme is False
    assert "Hawkish" in rel.policy_bias


def test_calendar_repository():
    cal = EconomicCalendar()
    releases = cal.list_releases()
    assert len(releases) >= 8

    # Check filtering
    cpi_releases = cal.list_releases(event_type=MacroEventType.CPI)
    assert len(cpi_releases) >= 4
    for r in cpi_releases:
        assert r.event_type == MacroEventType.CPI

    df = cal.to_dataframe()
    assert not df.empty
    assert "z_score" in df.columns
    assert "event_id" in df.columns
