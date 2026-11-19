"""
Unit and Integration tests for Historical Cohorts & End-to-End Orchestrator Pipeline.
"""

import os
import numpy as np
import pytest

from quantnexus_macro.engine import QuantNexusEngine
from quantnexus_macro.config import MacroEventType


def test_confidence_cone_properties():
    engine = QuantNexusEngine(cache_dir=None, reports_dir="reports_test")
    cone = engine.cohort_analyzer.build_confidence_cone(
        event_type=MacroEventType.CPI,
        surprise_sign=1,
        ticker="SPY",
        pre_minutes=20,
        post_minutes=40,
    )

    assert cone.sample_size >= 3
    assert len(cone.p10) == (20 + 40 + 1)
    assert len(cone.p90) == (20 + 40 + 1)

    # Monotonic quantile property: p10 <= p25 <= p50 <= p75 <= p90
    assert np.all(cone.p10 <= cone.p25 + 1e-6)
    assert np.all(cone.p25 <= cone.p50_median + 1e-6)
    assert np.all(cone.p50_median <= cone.p75 + 1e-6)
    assert np.all(cone.p75 <= cone.p90 + 1e-6)


def test_end_to_end_engine_pipeline():
    engine = QuantNexusEngine(cache_dir=None, reports_dir="reports_test")

    pkg = engine.analyze_event(
        event_or_id="CPI_2024_02_13",
        tickers=["SPY", "TLT", "UUP", "SHY", "IEF"],
        pre_event_minutes=20,
        post_event_minutes=40,
        force_simulation=True,
        generate_report=True,
    )

    assert pkg.event.event_id == "CPI_2024_02_13"
    assert len(pkg.windows) == 5
    assert len(pkg.event_study_metrics) == 5
    assert pkg.curve_result is not None
    assert pkg.regime_result is not None
    assert pkg.dislocation_result is not None
    assert pkg.cohort_benchmark is not None
    assert pkg.report_html_path is not None
    assert os.path.exists(pkg.report_html_path)

    # Clean up test report file
    if os.path.exists(pkg.report_html_path):
        os.remove(pkg.report_html_path)
    if os.path.exists("reports_test"):
        try:
            os.rmdir("reports_test")
        except Exception:
            pass
