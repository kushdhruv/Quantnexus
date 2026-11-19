"""
QuantNexus-Macro: Institutional Interactive Plotly Visual Suite
Multi-panel dark-themed dashboard featuring:
1. Cross-Asset T0 Normalized Trajectories
2. Multi-Tenor Yield Curve Twist & Slope Evolution
3. Macro Dislocation Index (MDI) Barometer
4. Empirical Confidence Fan Chart (Cohort Quantiles)
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quantnexus_macro.core.calendar import MacroRelease
from quantnexus_macro.core.data_feed import MarketWindow
from quantnexus_macro.analytics.curve import CurveAnalysisResult
from quantnexus_macro.analytics.dislocation import DislocationMatrixResult
from quantnexus_macro.analytics.cohort_backtest import CohortBenchmarkResult


class InstitutionalDashboard:
    """
    Constructs high-resolution, institutional-grade Plotly visual figures
    optimized for prop trading desks and executive macro presentations.
    """

    COLOR_PALETTE = {
        # Equities
        "SPY": "#00E5FF",  # Bright Cyan
        "QQQ": "#7C4DFF",  # Deep Purple
        "IWM": "#FF4081",  # Vivid Pink
        # Rates
        "SHY": "#448AFF",  # Soft Blue
        "IEI": "#2979FF",  # Medium Blue
        "IEF": "#2962FF",  # Royal Blue
        "TLT": "#304FFE",  # Indigo
        # FX
        "UUP": "#00E676",  # Emerald Green
        "FXE": "#1DE9B6",  # Teal Green
        "FXY": "#69F0AE",  # Mint Green
        # Commodities
        "GLD": "#FFD700",  # Gold
        "USO": "#FF9100",  # Amber / Orange
        # Credit
        "HYG": "#FF5252",  # Coral Red
    }

    def create_master_figure(
        self,
        event: MacroRelease,
        windows: Dict[str, MarketWindow],
        curve_result: Optional[CurveAnalysisResult] = None,
        dislocation_result: Optional[DislocationMatrixResult] = None,
        cohort_result: Optional[CohortBenchmarkResult] = None,
    ) -> go.Figure:
        """
        Creates a master 4-quadrant institutional dashboard figure:
        [Row 1, Col 1]: Cross-Asset Normalized T0 Response (Indexed to 100)
        [Row 1, Col 2]: Multi-Tenor Yield Shifts & Curve Twist (bps)
        [Row 2, Col 1]: Macro Dislocation Index (MDI Z-score Barometer)
        [Row 2, Col 2]: Historical Cohort Fan Chart & Empirical Quantile Cone
        """
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                f"<b>Cross-Asset Transmission ($T_0=100$)</b> | Shock Z: {event.z_score:+.2f}σ",
                "<b>Yield Curve Level Shifts & Twist</b> (Basis Points)",
                "<b>Macro Dislocation Index (MDI)</b> | Statistical Arbitrage Edge",
                f"<b>Historical Cohort Fan Chart</b> | {cohort_result.ticker if cohort_result else 'Benchmark'}",
            ),
            vertical_spacing=0.12,
            horizontal_spacing=0.08,
        )

        # -------------------------------------------------------------
        # PANEL 1: Cross-Asset Normalized Trajectories
        # -------------------------------------------------------------
        for ticker, w in windows.items():
            color = self.COLOR_PALETTE.get(ticker, "#FFFFFF")
            rel_min = w.relative_minutes
            norm = w.normalized_prices.values

            # Line styling based on asset class
            line_dash = "solid"
            if ticker in ("SHY", "IEI", "IEF", "TLT"):
                line_dash = "dash"
            elif ticker in ("UUP", "FXE", "FXY"):
                line_dash = "dot"

            fig.add_trace(
                go.Scatter(
                    x=rel_min,
                    y=norm,
                    mode="lines",
                    name=f"{ticker} ({w.meta.name.split('(')[0].strip()})",
                    line=dict(color=color, width=2, dash=line_dash),
                    hovertemplate=(
                        f"<b>{ticker}</b><br>"
                        "Rel Minute: %{x}m<br>"
                        "Normalized: %{y:.2f}<br>"
                        "Return from T0: %{customdata:.2f}%<extra></extra>"
                    ),
                    customdata=norm - 100.0,
                ),
                row=1,
                col=1,
            )

        # Vertical T0 release marker
        fig.add_vline(x=0, line_dash="solid", line_color="#FF1744", line_width=2, row=1, col=1)
        fig.add_annotation(
            x=0,
            y=100.0,
            text="T0 Release",
            showarrow=True,
            arrowhead=2,
            arrowcolor="#FF1744",
            font=dict(color="#FF1744", size=10),
            row=1,
            col=1,
        )

        # -------------------------------------------------------------
        # PANEL 2: Yield Curve Shifts (Basis Points)
        # -------------------------------------------------------------
        if curve_result:
            tenors = [2, 5, 10, 20]
            tenor_labels = ["2Y (SHY)", "5Y (IEI)", "10Y (IEF)", "20Y+ (TLT)"]
            bps_shifts = [
                curve_result.tenor_shifts.get(t, None).yield_change_bps
                for t in tenors
                if t in curve_result.tenor_shifts
            ]
            colors = ["#FF5252" if b >= 0 else "#00E676" for b in bps_shifts]

            fig.add_trace(
                go.Bar(
                    x=tenor_labels[:len(bps_shifts)],
                    y=bps_shifts,
                    marker=dict(color=colors),
                    name="Yield Shift (bps)",
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>Yield Shift: %{y:+.2f} bps<extra></extra>",
                ),
                row=1,
                col=2,
            )
            # Add subtitle annotation for Curve Regime
            fig.add_annotation(
                xref="x2 domain",
                yref="y2 domain",
                x=0.5,
                y=1.08,
                text=f"<b>Regime:</b> {curve_result.regime.value.split('(')[0].strip()} | 2s10s: {curve_result.slope_2s10s_bps_change:+.1f} bps",
                showarrow=False,
                font=dict(color="#FFD700", size=11),
                row=1,
                col=2,
            )

        # -------------------------------------------------------------
        # PANEL 3: Macro Dislocation Index (MDI Barometer)
        # -------------------------------------------------------------
        if dislocation_result:
            sig_list = dislocation_result.signals
            sig_tickers = [s.ticker for s in sig_list]
            mdis = [s.macro_dislocation_index for s in sig_list]

            bar_colors = []
            for m in mdis:
                if m >= 1.65:
                    bar_colors.append("#FF1744")  # Red: Overextended / Lagging Short
                elif m <= -1.65:
                    bar_colors.append("#00E676")  # Green: Oversold / Lagging Long
                else:
                    bar_colors.append("#78909C")  # Grey: Fairly priced

            fig.add_trace(
                go.Bar(
                    x=sig_tickers,
                    y=mdis,
                    marker=dict(color=bar_colors),
                    name="MDI (Z-score)",
                    showlegend=False,
                    hovertemplate="<b>%{x}</b><br>MDI: %{y:+.2f}σ<extra></extra>",
                ),
                row=2,
                col=1,
            )

            # Statistical threshold bands (+/- 1.65 sigma)
            fig.add_hline(y=1.65, line_dash="dash", line_color="#FF5252", line_width=1, row=2, col=1)
            fig.add_hline(y=-1.65, line_dash="dash", line_color="#69F0AE", line_width=1, row=2, col=1)

        # -------------------------------------------------------------
        # PANEL 4: Historical Cohort Fan Chart / Confidence Cone
        # -------------------------------------------------------------
        if cohort_result:
            cone = cohort_result.cone
            x_m = cone.relative_minutes

            # Shaded bands: 10th to 90th percentile
            fig.add_trace(
                go.Scatter(
                    x=list(x_m) + list(x_m[::-1]),
                    y=list(cone.p90) + list(cone.p10[::-1]),
                    fill="toself",
                    fillcolor="rgba(33, 150, 243, 0.15)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="10-90th Percentile Band",
                ),
                row=2,
                col=2,
            )

            # Shaded bands: 25th to 75th percentile
            fig.add_trace(
                go.Scatter(
                    x=list(x_m) + list(x_m[::-1]),
                    y=list(cone.p75) + list(cone.p25[::-1]),
                    fill="toself",
                    fillcolor="rgba(33, 150, 243, 0.25)",
                    line=dict(color="rgba(255,255,255,0)"),
                    hoverinfo="skip",
                    showlegend=False,
                    name="25-75th Percentile Band",
                ),
                row=2,
                col=2,
            )

            # Median historical trajectory
            fig.add_trace(
                go.Scatter(
                    x=x_m,
                    y=cone.p50_median,
                    mode="lines",
                    line=dict(color="#64B5F6", width=1.5, dash="dot"),
                    name="Cohort Median",
                ),
                row=2,
                col=2,
            )

            # Current live trajectory
            fig.add_trace(
                go.Scatter(
                    x=x_m,
                    y=cohort_result.current_trajectory,
                    mode="lines",
                    line=dict(color="#FFD700", width=3),
                    name=f"Current {cohort_result.ticker}",
                ),
                row=2,
                col=2,
            )

            fig.add_vline(x=0, line_dash="solid", line_color="#FF1744", row=2, col=2)

        # Global Layout Styling
        fig.update_layout(
            title=dict(
                text=(
                    f"<b>QuantNexus-Macro Terminal</b> | {event.name} "
                    f"({event.timestamp.strftime('%Y-%m-%d %H:%M UTC')}) | "
                    f"Actual: {event.actual}{event.unit} vs Cons: {event.consensus}{event.unit} "
                    f"| Surprise: {event.surprise:+.2f}{event.unit} ({event.z_score:+.2f}σ)"
                ),
                font=dict(size=14, color="#E0E0E0"),
            ),
            template="plotly_dark",
            paper_bgcolor="#121824",
            plot_bgcolor="#182232",
            height=900,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
                font=dict(size=10),
            ),
        )

        fig.update_xaxes(title_text="Minutes from T0 Release", gridcolor="#223046", row=1, col=1)
        fig.update_yaxes(title_text="Indexed to 100 at T0", gridcolor="#223046", row=1, col=1)

        fig.update_xaxes(title_text="Bond Tenor", gridcolor="#223046", row=1, col=2)
        fig.update_yaxes(title_text="Yield Shift (bps)", gridcolor="#223046", row=1, col=2)

        fig.update_xaxes(title_text="Asset Ticker", gridcolor="#223046", row=2, col=1)
        fig.update_yaxes(title_text="MDI (Standardized Z)", gridcolor="#223046", row=2, col=1)

        fig.update_xaxes(title_text="Minutes from T0 Release", gridcolor="#223046", row=2, col=2)
        fig.update_yaxes(title_text="Indexed to 100", gridcolor="#223046", row=2, col=2)

        return fig
