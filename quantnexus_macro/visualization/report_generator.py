"""
QuantNexus-Macro: Standalone Institutional HTML Report Generator
Assembles interactive Plotly dashboards, KPI metric banners, econometric tables,
and execution blotters into a self-contained, publication-grade analytical dossier.
"""

import os
from typing import Dict, List, Optional
import pandas as pd

from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import MarketWindow
from quantnexus_macro.core.event_study import EventStudyMetrics
from quantnexus_macro.analytics.curve import CurveAnalysisResult
from quantnexus_macro.analytics.dislocation import DislocationMatrixResult
from quantnexus_macro.analytics.regime import RegimeClassificationResult
from quantnexus_macro.analytics.cohort_backtest import CohortBenchmarkResult
from quantnexus_macro.analytics.lead_lag import AbsorptionVelocityResult, LeadLagPairResult
from quantnexus_macro.signals.execution_signals import ExecutionTicket
from quantnexus_macro.visualization.dashboard import InstitutionalDashboard


class MacroReportGenerator:
    """
    Generates standalone, institutional HTML dossiers with zero external server dependencies.
    """

    def __init__(self, output_dir: str = "reports"):
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.dashboard = InstitutionalDashboard()

    def generate_html_report(
        self,
        event: MacroRelease,
        windows: Dict[str, MarketWindow],
        study_metrics: Dict[str, EventStudyMetrics],
        curve_result: Optional[CurveAnalysisResult],
        regime_result: Optional[RegimeClassificationResult],
        dislocation_result: Optional[DislocationMatrixResult],
        cohort_result: Optional[CohortBenchmarkResult],
        lead_lag_velocities: Optional[Dict[str, AbsorptionVelocityResult]] = None,
        tickets: Optional[List[ExecutionTicket]] = None,
        filename: Optional[str] = None,
    ) -> str:
        """Render complete HTML dossier and write to disk."""
        fig = self.dashboard.create_master_figure(
            event=event,
            windows=windows,
            curve_result=curve_result,
            dislocation_result=dislocation_result,
            cohort_result=cohort_result,
        )

        chart_html = fig.to_html(full_html=False, include_plotlyjs="cdn")

        # Filename
        if filename is None:
            clean_id = event.event_id.replace(" ", "_")
            filename = f"quantnexus_macro_{clean_id}.html"

        filepath = os.path.join(self.output_dir, filename)

        # Build tables HTML
        metrics_table_html = self._build_event_study_table_html(study_metrics)
        signals_table_html = self._build_signals_table_html(dislocation_result.signals if dislocation_result else [])
        tickets_table_html = self._build_tickets_table_html(tickets or [])
        lead_lag_table_html = self._build_lead_lag_table_html(lead_lag_velocities or {})

        # Top Opportunity banner
        top_opp = dislocation_result.highest_alpha_opportunity if dislocation_result else None
        top_opp_text = (
            f"<b>{top_opp.action.value}</b> on <b>{top_opp.ticker}</b> | Target: {top_opp.target_price:.2f} | MDI: {top_opp.macro_dislocation_index:+.2f}σ"
            if top_opp else "No statistically significant dislocation (> 1.65σ) detected."
        )

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>QuantNexus-Macro | {event.name} Dossier</title>
    <style>
        body {{
            background-color: #0b0f19;
            color: #d1d5db;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 24px;
        }}
        .header {{
            background: linear-gradient(135deg, #131c2e, #1e293b);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 24px;
            margin-bottom: 24px;
        }}
        .header h1 {{
            margin: 0 0 8px 0;
            font-size: 24px;
            color: #38bdf8;
            letter-spacing: -0.5px;
        }}
        .header .subtitle {{
            color: #94a3b8;
            font-size: 14px;
            margin-bottom: 16px;
        }}
        .kpi-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .kpi-card {{
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 14px;
        }}
        .kpi-title {{
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            color: #64748b;
            margin-bottom: 6px;
        }}
        .kpi-value {{
            font-size: 18px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .kpi-sub {{
            font-size: 12px;
            color: #94a3b8;
            margin-top: 4px;
        }}
        .section {{
            background-color: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 24px;
        }}
        .section-title {{
            font-size: 18px;
            font-weight: 600;
            color: #38bdf8;
            margin-top: 0;
            margin-bottom: 16px;
            border-bottom: 1px solid #1e293b;
            padding-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin-top: 8px;
        }}
        th {{
            background-color: #1e293b;
            color: #94a3b8;
            text-align: left;
            padding: 10px 12px;
            font-weight: 600;
        }}
        td {{
            padding: 9px 12px;
            border-bottom: 1px solid #1e293b;
        }}
        tr:hover td {{
            background-color: #1a2438;
        }}
        .badge-buy {{
            background-color: #064e3b;
            color: #34d399;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
        }}
        .badge-sell {{
            background-color: #4c0519;
            color: #fb7185;
            padding: 3px 8px;
            border-radius: 4px;
            font-weight: 600;
            font-size: 11px;
        }}
        .badge-neutral {{
            background-color: #1e293b;
            color: #94a3b8;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}
        .footer {{
            text-align: center;
            font-size: 12px;
            color: #64748b;
            margin-top: 40px;
        }}
    </style>
</head>
<body>

    <div class="header">
        <h1>QuantNexus-Macro Terminal | Event Transmission Dossier</h1>
        <div class="subtitle">
            <b>{event.name}</b> | Released: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')} |
            Actual: <b>{event.actual}{event.unit}</b> vs Consensus: <b>{event.consensus}{event.unit}</b>
            (Surprise: <b>{event.surprise:+.2f}{event.unit}</b> | Z: <b>{event.z_score:+.2f}σ</b>)
        </div>

        <div class="kpi-grid">
            <div class="kpi-card">
                <div class="kpi-title">Detected Macro Regime</div>
                <div class="kpi-value" style="color: #38bdf8;">{regime_result.detected_regime.name if regime_result else 'N/A'}</div>
                <div class="kpi-sub">Confidence: {int((regime_result.confidence_score if regime_result else 0.5) * 100)}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Yield Curve Twist (2s10s)</div>
                <div class="kpi-value" style="color: {'#fb7185' if curve_result and curve_result.slope_2s10s_bps_change < 0 else '#34d399'};">
                    {curve_result.slope_2s10s_bps_change:+.1f} bps
                </div>
                <div class="kpi-sub">{curve_result.regime.name.replace('_', ' ') if curve_result else 'N/A'}</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Market Efficiency Score</div>
                <div class="kpi-value">{dislocation_result.cross_asset_efficiency_score if dislocation_result else 100.0}%</div>
                <div class="kpi-sub">Mean Disloc: {dislocation_result.mean_absolute_dislocation if dislocation_result else 0.0}%</div>
            </div>
            <div class="kpi-card">
                <div class="kpi-title">Highest Alpha Opportunity</div>
                <div class="kpi-value" style="font-size: 14px; color: #f59e0b;">{top_opp.ticker if top_opp else 'None'}</div>
                <div class="kpi-sub">{top_opp.action.name.replace('_', ' ') if top_opp else 'Fairly priced'}</div>
            </div>
        </div>
    </div>

    <!-- Chart Panel -->
    <div class="section">
        <div class="section-title">Institutional Interactive Visual Suite</div>
        {chart_html}
    </div>

    <!-- Signals & Execution Blotter -->
    <div class="section">
        <div class="section-title">Macro Transmission Dislocation (MTDE) & Alpha Signals</div>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 0;">
            Identifies assets lagging behind theoretical macro elasticity or over-reacting due to liquidity exhaustion.
        </p>
        {signals_table_html}
    </div>

    <!-- Execution Tickets -->
    <div class="section">
        <div class="section-title">Prop Desk Algorithmic Execution Blotter</div>
        <p style="font-size: 13px; color: #94a3b8; margin-top: 0;">
            Position sizing sized to risk budget ($10M book with 15 bps max loss limit per ticket).
        </p>
        {tickets_table_html}
    </div>

    <!-- Econometric Event Study -->
    <div class="section">
        <div class="section-title">Econometric Event Study Metrics (Campbell-Lo-MacKinlay Framework)</div>
        {metrics_table_html}
    </div>

    <!-- Lead-Lag & Absorption Speeds -->
    <div class="section">
        <div class="section-title">Transmission Cascade & Information Absorption Speed</div>
        {lead_lag_table_html}
    </div>

    <div class="footer">
        QuantNexus-Macro Quantitative Systems &copy; 2026. For Institutional Quantitative Research & Proprietary Trading Use.
    </div>

</body>
</html>
"""
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return os.path.abspath(filepath)

    def _build_event_study_table_html(self, metrics: Dict[str, EventStudyMetrics]) -> str:
        rows = []
        for m in sorted(metrics.values(), key=lambda x: x.post_event_return_pct, reverse=True):
            ret_color = "#34d399" if m.post_event_return_pct >= 0 else "#fb7185"
            sig_badge = '<span class="badge-buy">YES</span>' if m.is_statistically_significant else '<span class="badge-neutral">No</span>'
            rows.append(f"""
                <tr>
                    <td><b>{m.ticker}</b></td>
                    <td>{m.name}</td>
                    <td>${m.t0_price:.2f}</td>
                    <td>${m.final_price:.2f}</td>
                    <td style="color: {ret_color}; font-weight: 600;">{m.post_event_return_pct:+.2f}%</td>
                    <td>{m.car_pct:+.2f}%</td>
                    <td style="color: #34d399;">+{m.max_favorable_excursion_pct:.2f}%</td>
                    <td style="color: #fb7185;">{m.max_adverse_excursion_pct:.2f}%</td>
                    <td>{m.vol_expansion_ratio:.2f}x</td>
                    <td>{m.t_statistic:+.2f}</td>
                    <td>{m.p_value:.4f}</td>
                    <td>{sig_badge}</td>
                </tr>
            """)
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Name</th>
                        <th>T0 Price</th>
                        <th>Final Price</th>
                        <th>Post Return</th>
                        <th>CAR</th>
                        <th>MFE</th>
                        <th>MAE</th>
                        <th>Vol Expansion</th>
                        <th>t-stat</th>
                        <th>p-val</th>
                        <th>Significant</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _build_signals_table_html(self, signals: List[DislocationSignal]) -> str:
        if not signals:
            return "<p>No signals generated.</p>"
        rows = []
        for s in signals:
            if "BUY" in s.action.value:
                badge = f'<span class="badge-buy">{s.action.value.split("(")[0].strip()}</span>'
            elif "SELL" in s.action.value:
                badge = f'<span class="badge-sell">{s.action.value.split("(")[0].strip()}</span>'
            else:
                badge = f'<span class="badge-neutral">NEUTRAL</span>'

            rows.append(f"""
                <tr>
                    <td><b>{s.ticker}</b></td>
                    <td>{s.name}</td>
                    <td>{badge}</td>
                    <td>${s.current_price:.2f}</td>
                    <td>${s.implied_fair_value_price:.2f}</td>
                    <td style="font-weight: 600; color: {'#34d399' if s.dislocation_pct < 0 else '#fb7185'};">{s.dislocation_pct:+.2f}%</td>
                    <td><b>{s.macro_dislocation_index:+.2f}σ</b></td>
                    <td>${s.target_price:.2f}</td>
                    <td>${s.stop_loss_price:.2f}</td>
                    <td>{s.risk_reward_ratio:.2f}</td>
                    <td>{s.half_life_minutes:.0f}m</td>
                    <td>{s.confidence_pct:.0f}%</td>
                    <td style="font-size: 11px; color: #94a3b8;">{s.rationale}</td>
                </tr>
            """)
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Name</th>
                        <th>Signal</th>
                        <th>Current Px</th>
                        <th>Fair Px</th>
                        <th>Dislocation</th>
                        <th>MDI (Z)</th>
                        <th>Target Px</th>
                        <th>Stop Px</th>
                        <th>R:R</th>
                        <th>Half-Life</th>
                        <th>Win Prob</th>
                        <th>Trading Rationale</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _build_tickets_table_html(self, tickets: List[ExecutionTicket]) -> str:
        if not tickets:
            return "<p>No active execution tickets.</p>"
        rows = []
        for t in tickets:
            act_badge = f'<span class="{"badge-buy" if t.action == "BUY" else "badge-sell"}">{t.action}</span>'
            hedge_str = f"{t.hedged_action} {t.hedged_ticker} ({t.hedge_ratio:.1f}x)" if t.hedged_ticker else "None"
            rows.append(f"""
                <tr>
                    <td><b>{t.ticket_id}</b></td>
                    <td>{t.strategy_type}</td>
                    <td>{act_badge} <b>{t.primary_ticker}</b></td>
                    <td>${t.entry_price:.2f}</td>
                    <td>${t.target_price:.2f}</td>
                    <td>${t.stop_loss:.2f}</td>
                    <td>{hedge_str}</td>
                    <td><b>${t.suggested_notional_usd:,.0f}</b></td>
                    <td style="color: #34d399;">+${t.expected_pnl_usd:,.0f}</td>
                    <td style="color: #fb7185;">-${t.max_loss_usd:,.0f}</td>
                    <td style="font-size: 11px; color: #94a3b8;">{t.execution_instructions}</td>
                </tr>
            """)
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Ticket ID</th>
                        <th>Strategy</th>
                        <th>Order</th>
                        <th>Entry</th>
                        <th>Target</th>
                        <th>Stop</th>
                        <th>Hedge</th>
                        <th>Notional ($)</th>
                        <th>Target PnL</th>
                        <th>Max Risk</th>
                        <th>Execution Instructions</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """

    def _build_lead_lag_table_html(self, velocities: Dict[str, AbsorptionVelocityResult]) -> str:
        if not velocities:
            return "<p>No velocity data available.</p>"
        rows = []
        for v in velocities.values():
            rows.append(f"""
                <tr>
                    <td><b>{v.ticker}</b></td>
                    <td>{v.name}</td>
                    <td>{v.t50_absorption_minutes} mins</td>
                    <td><b>{v.t80_absorption_minutes} mins</b></td>
                    <td><span class="badge-buy">{v.absorption_speed}</span></td>
                </tr>
            """)
        return f"""
            <table>
                <thead>
                    <tr>
                        <th>Ticker</th>
                        <th>Name</th>
                        <th>T50 Absorption (50% of move)</th>
                        <th>T80 Absorption (80% of move)</th>
                        <th>Speed Classification</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """
