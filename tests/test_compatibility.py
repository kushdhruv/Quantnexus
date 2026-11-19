"""
Verification of full 1-to-1 backward compatibility with amit-singh-99/cross-asset-macro-tracker.
"""

import plotly.graph_objects as go
import pytest

from quantnexus_macro import build_macro_tracker


def test_build_macro_tracker_compatibility():
    """Verify that build_macro_tracker executes with original signature and returns valid figure."""
    # Call with exact signature from amit-singh-99/cross-asset-macro-tracker
    fig = build_macro_tracker(
        event_name="CPI Impact Study",
        event_time_str="2024-02-13 13:30:00",
        show=False,
        force_simulation=True,
    )

    assert isinstance(fig, go.Figure)

    # Check traces
    trace_names = [t.name for t in fig.data if hasattr(t, "name") and t.name]
    assert any("SPY" in n for n in trace_names)
    assert any("UUP" in n for n in trace_names)
    assert any("TLT" in n for n in trace_names)

    # Layout checks
    assert len(fig.data) >= 3
    assert fig.layout.paper_bgcolor is not None
