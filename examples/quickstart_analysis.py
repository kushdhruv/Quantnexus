"""
QuantNexus-Macro: Quickstart Example Script
Demonstrates end-to-end programmatic analysis of a macroeconomic release.
"""

import sys
import os

# Ensure current working directory is on path
sys.path.insert(0, os.path.abspath("."))

from quantnexus_macro.engine import QuantNexusEngine
from quantnexus_macro.config import MacroEventType


def run_quickstart():
    print("=" * 70)
    print("QuantNexus-Macro: Institutional Transmission & Dislocation Analysis")
    print("=" * 70)

    # 1. Initialize Engine
    engine = QuantNexusEngine(book_capital_usd=10_000_000.0)

    # 2. Inspect available calendar releases
    print("\n[1] Checking Calendar Benchmark Events...")
    releases = engine.calendar.list_releases(min_abs_z=1.0)
    for r in releases[:5]:
        print(f"  - {r.event_id}: {r.name} | Surprise: {r.surprise:+.2f}{r.unit} ({r.z_score:+.2f}σ) [{r.policy_bias}]")

    # 3. Analyze Target Shock: e.g. CPI 2024-02-13
    target_event_id = "CPI_2024_02_13"
    print(f"\n[2] Executing Full Cross-Asset Transmission Analysis for '{target_event_id}'...")

    pkg = engine.analyze_event(
        event_or_id=target_event_id,
        pre_event_minutes=30,
        post_event_minutes=120,
        benchmark_cohort_ticker="SPY",
        generate_report=True,
    )

    event = pkg.event
    print(f"\n[3] Event Summary:")
    print(f"    Release: {event.name}")
    print(f"    Actual: {event.actual}{event.unit} | Consensus: {event.consensus}{event.unit}")
    print(f"    Standardized Surprise: {event.surprise:+.2f}{event.unit} ({event.z_score:+.2f}σ)")
    print(f"    Detected Macro Regime: {pkg.regime_result.detected_regime.value}")

    # 4. Yield Curve Deconstruction
    if pkg.curve_result:
        c = pkg.curve_result
        print(f"\n[4] Yield Curve Deconstruction:")
        print(f"    Curve Regime: {c.regime.value}")
        print(f"    2s10s Slope Delta: {c.slope_2s10s_bps_change:+.1f} bps")
        print(f"    5s30s Slope Delta: {c.slope_5s30s_bps_change:+.1f} bps")
        for tenor, shift in c.tenor_shifts.items():
            print(f"    - Tenor {tenor:2d}Y ({shift.ticker}): Yield Shift: {shift.yield_change_bps:+.2f} bps (Price Ret: {shift.price_return_pct:+.2f}%)")

    # 5. [GAME CHANGER] Dislocation Signals
    d = pkg.dislocation_result
    print(f"\n[5] Macro Transmission Dislocation Engine (MTDE):")
    print(f"    Cross-Asset Efficiency Score: {d.cross_asset_efficiency_score}%")
    print(f"    Mean Absolute Dislocation: {d.mean_absolute_dislocation}%")

    top_opp = d.highest_alpha_opportunity
    if top_opp:
        print(f"\n    >>> TOP ALPHA OPPORTUNITY <<<")
        print(f"    Asset: {top_opp.ticker} ({top_opp.name})")
        print(f"    Signal Action: {top_opp.action.value}")
        print(f"    Current Price: ${top_opp.current_price:.2f} -> Theoretical Macro Target: ${top_opp.target_price:.2f}")
        print(f"    Dislocation: {top_opp.dislocation_pct:+.2f}% (MDI = {top_opp.macro_dislocation_index:+.2f}σ)")
        print(f"    Stop Loss: ${top_opp.stop_loss_price:.2f} | Risk/Reward: {top_opp.risk_reward_ratio:.2f}")
        print(f"    Expected Convergence Horizon: ~{top_opp.expected_convergence_minutes} minutes")
        print(f"    Rationale: {top_opp.rationale}")

    # 6. Algorithmic Execution Tickets
    print(f"\n[6] Execution Blotter ({len(pkg.execution_tickets)} tickets emitted):")
    for t in pkg.execution_tickets[:3]:
        print(f"    Ticket {t.ticket_id}: {t.action} {t.primary_ticker} | Notional: ${t.suggested_notional_usd:,.0f} | Target PnL: +${t.expected_pnl_usd:,.0f} | Max Loss: -${t.max_loss_usd:,.0f}")

    print(f"\n[7] Interactive HTML Dossier generated at:")
    print(f"    {pkg.report_html_path}")
    print("=" * 70)


if __name__ == "__main__":
    run_quickstart()
