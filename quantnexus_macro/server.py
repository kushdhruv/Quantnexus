"""
QuantNexus-Macro: Interactive Web & REST API Service
Exposes JSON endpoints and serves real-time dashboards for prop desk algorithmic trading integration.
"""

from typing import Dict, List, Optional
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from quantnexus_macro.engine import QuantNexusEngine
from quantnexus_macro.config import DEFAULT_ASSET_UNIVERSE

app = FastAPI(
    title="QuantNexus-Macro Institutional API",
    description="Cross-Asset Macroeconomic Transmission & Dislocation Engine",
    version="1.0.0",
)

engine = QuantNexusEngine()


@app.get("/", response_class=HTMLResponse)
def index():
    """Welcome portal with quick links to events and live reports."""
    releases = engine.calendar.list_releases()
    rows = []
    for r in releases:
        rows.append(
            f"<li><b><a href='/api/report/{r.event_id}'>{r.name}</a></b> "
            f"({r.timestamp.strftime('%Y-%m-%d %H:%M UTC')}) - Surprise: {r.surprise:+.2f}{r.unit} ({r.z_score:+.2f}σ) - "
            f"<a href='/api/analyze/{r.event_id}'>[JSON API]</a></li>"
        )

    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>QuantNexus-Macro Terminal</title>
        <style>
            body {{ background-color: #0b0f19; color: #d1d5db; font-family: sans-serif; padding: 40px; }}
            h1 {{ color: #38bdf8; }}
            a {{ color: #38bdf8; text-decoration: none; }}
            a:hover {{ text-decoration: underline; }}
            li {{ margin-bottom: 12px; }}
            .card {{ background: #1e293b; padding: 20px; border-radius: 8px; max-width: 800px; }}
        </style>
    </head>
    <body>
        <h1>QuantNexus-Macro Institutional Terminal</h1>
        <p>Real-Time Macroeconomic Transmission, Yield Curve Twist & Dislocation Engine</p>
        <div class="card">
            <h3>Available Macro Benchmark Events</h3>
            <ul>
                {''.join(rows)}
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/api/events")
def get_calendar_events():
    """Retrieve all registered macroeconomic release events."""
    return engine.calendar.to_dataframe().to_dict(orient="records")


@app.get("/api/analyze/{event_id}")
def analyze_event_endpoint(
    event_id: str,
    pre: int = Query(30, description="Pre-event window in minutes"),
    post: int = Query(120, description="Post-event window in minutes"),
    sim: bool = Query(False, description="Force calibrated simulation"),
):
    """Run full quantitative macro transmission and dislocation analysis."""
    try:
        pkg = engine.analyze_event(
            event_or_id=event_id,
            pre_event_minutes=pre,
            post_event_minutes=post,
            force_simulation=sim,
            generate_report=True,
        )

        return {
            "event": {
                "id": pkg.event.event_id,
                "name": pkg.event.name,
                "timestamp": pkg.event.timestamp.isoformat(),
                "actual": pkg.event.actual,
                "consensus": pkg.event.consensus,
                "surprise": pkg.event.surprise,
                "z_score": pkg.event.z_score,
                "policy_bias": pkg.event.policy_bias,
            },
            "regime": {
                "detected": pkg.regime_result.detected_regime.value,
                "confidence": pkg.regime_result.confidence_score,
                "rate_dir": pkg.regime_result.rate_direction,
                "usd_dir": pkg.regime_result.usd_direction,
                "equity_dir": pkg.regime_result.equity_direction,
                "narrative": pkg.regime_result.narrative_summary,
            },
            "curve": {
                "regime": pkg.curve_result.regime.value if pkg.curve_result else None,
                "slope_2s10s_bps": pkg.curve_result.slope_2s10s_bps_change if pkg.curve_result else None,
                "slope_5s30s_bps": pkg.curve_result.slope_5s30s_bps_change if pkg.curve_result else None,
            },
            "dislocations": {
                "efficiency_score": pkg.dislocation_result.cross_asset_efficiency_score,
                "mean_abs_dislocation": pkg.dislocation_result.mean_absolute_dislocation,
                "signals": [
                    {
                        "ticker": s.ticker,
                        "action": s.action.value,
                        "current_px": s.current_price,
                        "fair_px": s.implied_fair_value_price,
                        "dislocation_pct": s.dislocation_pct,
                        "mdi_z": s.macro_dislocation_index,
                        "target_px": s.target_price,
                        "stop_px": s.stop_loss_price,
                        "risk_reward": s.risk_reward_ratio,
                        "confidence_pct": s.confidence_pct,
                        "rationale": s.rationale,
                    }
                    for s in pkg.dislocation_result.signals
                ],
            },
            "execution_tickets": [
                {
                    "ticket_id": t.ticket_id,
                    "strategy": t.strategy_type,
                    "action": t.action,
                    "ticker": t.primary_ticker,
                    "entry": t.entry_price,
                    "target": t.target_price,
                    "stop": t.stop_loss,
                    "hedged_ticker": t.hedged_ticker,
                    "hedged_action": t.hedged_action,
                    "notional_usd": t.suggested_notional_usd,
                    "expected_pnl": t.expected_pnl_usd,
                    "max_loss": t.max_loss_usd,
                }
                for t in pkg.execution_tickets
            ],
            "report_path": pkg.report_html_path,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/report/{event_id}", response_class=HTMLResponse)
def get_report_html(event_id: str):
    """Generate and return standalone HTML report for the event."""
    try:
        pkg = engine.analyze_event(event_or_id=event_id, generate_report=True)
        if pkg.report_html_path and os.path.exists(pkg.report_html_path):
            with open(pkg.report_html_path, "r", encoding="utf-8") as f:
                return HTMLResponse(content=f.read())
        raise HTTPException(status_code=404, detail="Report generation failed.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
