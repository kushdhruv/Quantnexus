"""
QuantNexus-Macro: Multi-Tenor Yield Curve & Curve Spread Analytics
Calculates yield curve level shifts, slope dynamics (2s10s, 5s30s), curvature (butterfly),
and institutional curve twist regimes (Bear Flattener, Bull Steepener, etc.).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum
import numpy as np
import pandas as pd

from quantnexus_macro.core.data_feed import MarketWindow


class CurveRegime(str, Enum):
    BEAR_FLATTENER = "Bear Flattener (Front-End Yields Spike > Long-End; Hawkish Shock)"
    BEAR_STEEPENER = "Bear Steepener (Long-End Yields Spike > Front-End; Term Premium Shock)"
    BULL_FLATTENER = "Bull Flattener (Long-End Yields Drop > Front-End; Growth Slowdown)"
    BULL_STEEPENER = "Bull Steepener (Front-End Yields Collapse > Long-End; Rapid Easing)"
    PARALLEL_SHIFT_UP = "Parallel Shift Up (Rates Higher Uniformly)"
    PARALLEL_SHIFT_DOWN = "Parallel Shift Down (Rates Lower Uniformly)"
    NEUTRAL_TWIST = "Unchanged / Negligible Curve Shift"


@dataclass
class YieldShiftResult:
    tenor_years: int
    ticker: str
    duration_proxy: float
    price_return_pct: float
    yield_change_bps: float  # Estimated basis point yield change: -(dP/P) / D * 10,000


@dataclass
class CurveAnalysisResult:
    tenor_shifts: Dict[int, YieldShiftResult]
    slope_2s10s_bps_change: float
    slope_5s30s_bps_change: float
    butterfly_2s5s10s_bps_change: float
    regime: CurveRegime
    regime_rationale: str
    intraday_slope_series: pd.Series  # Minute-by-minute 2s10s yield spread evolution (bps)


class YieldCurveAnalyzer:
    """
    Fixed income yield curve decomposition engine.
    Maps ETF price changes back to implied benchmark bond yield moves and curve twists.
    """

    def __init__(self):
        # Tenor mapping to ticker and duration proxies
        self.tenor_map = {
            2: {"ticker": "SHY", "duration": 1.90},
            5: {"ticker": "IEI", "duration": 4.50},
            10: {"ticker": "IEF", "duration": 7.80},
            20: {"ticker": "TLT", "duration": 16.50},
        }

    def analyze_curve_response(
        self, windows: Dict[str, MarketWindow]
    ) -> Optional[CurveAnalysisResult]:
        """Deconstruct multi-tenor bond price changes into curve twist and slope dynamics."""
        shifts: Dict[int, YieldShiftResult] = {}

        for tenor, info in self.tenor_map.items():
            ticker = info["ticker"]
            duration = info["duration"]
            if ticker not in windows:
                continue

            w = windows[ticker]
            t0_price = w.t0_price
            final_price = float(w.prices.iloc[-1])
            price_ret = (final_price / t0_price) - 1.0

            # Delta Yield (bps) = -(Delta P / P) / Duration * 10,000
            # Example: -1.0% price move with Duration 7.8 -> +12.82 bps yield move
            yield_bps = -(price_ret / duration) * 10000.0

            shifts[tenor] = YieldShiftResult(
                tenor_years=tenor,
                ticker=ticker,
                duration_proxy=duration,
                price_return_pct=round(price_ret * 100.0, 2),
                yield_change_bps=round(yield_bps, 2),
            )

        if 2 not in shifts or 10 not in shifts:
            return None

        # 2s10s Slope Delta = Delta Y_10 - Delta Y_2
        dy2 = shifts[2].yield_change_bps
        dy10 = shifts[10].yield_change_bps
        slope_2s10s_delta = dy10 - dy2

        # 5s30s Slope Delta = Delta Y_20 - Delta Y_5 (using 20Y TLT as long proxy)
        dy5 = shifts.get(5, shifts[2]).yield_change_bps
        dy20 = shifts.get(20, shifts[10]).yield_change_bps
        slope_5s30s_delta = dy20 - dy5

        # Butterfly Curvature Delta = 2*Y_5 - (Y_2 + Y_10)
        butterfly_delta = 2.0 * dy5 - (dy2 + dy10)

        # Classify curve regime
        regime, rationale = self._classify_curve_regime(dy2, dy10, slope_2s10s_delta)

        # Intraday 2s10s slope series
        intraday_slope = self._compute_intraday_slope_series(
            windows["SHY"], windows["IEF"], self.tenor_map[2]["duration"], self.tenor_map[10]["duration"]
        )

        return CurveAnalysisResult(
            tenor_shifts=shifts,
            slope_2s10s_bps_change=round(slope_2s10s_delta, 2),
            slope_5s30s_bps_change=round(slope_5s30s_delta, 2),
            butterfly_2s5s10s_bps_change=round(butterfly_delta, 2),
            regime=regime,
            regime_rationale=rationale,
            intraday_slope_series=intraday_slope,
        )

    def _classify_curve_regime(
        self, dy2: float, dy10: float, slope_delta: float
    ) -> Tuple[CurveRegime, str]:
        """Classify curve twist using standard institutional rates taxonomy."""
        threshold = 1.0  # bps minimum shift

        if abs(dy2) < threshold and abs(dy10) < threshold:
            return CurveRegime.NEUTRAL_TWIST, "Negligible yield shift (< 1.0 bps across curve)"

        if dy2 > 0 and dy10 > 0:
            if slope_delta <= -1.0:
                return (
                    CurveRegime.BEAR_FLATTENER,
                    f"2Y yield rose {dy2:+.1f} bps, outpacing 10Y (+{dy10:+.1f} bps). "
                    f"2s10s slope flattened by {slope_delta:.1f} bps (Classic Hawkish repricing).",
                )
            elif slope_delta >= 1.0:
                return (
                    CurveRegime.BEAR_STEEPENER,
                    f"10Y yield rose {dy10:+.1f} bps, outpacing 2Y (+{dy2:+.1f} bps). "
                    f"2s10s slope steepened by {slope_delta:+.1f} bps (Term premium / supply expansion).",
                )
            else:
                return (
                    CurveRegime.PARALLEL_SHIFT_UP,
                    f"Uniform upward shift across curve (2Y: {dy2:+.1f} bps, 10Y: {dy10:+.1f} bps).",
                )

        elif dy2 < 0 and dy10 < 0:
            if slope_delta >= 1.0:
                return (
                    CurveRegime.BULL_STEEPENER,
                    f"2Y yield dropped {dy2:.1f} bps, faster than 10Y ({dy10:.1f} bps). "
                    f"2s10s slope steepened by {slope_delta:+.1f} bps (Aggressive central bank easing).",
                )
            elif slope_delta <= -1.0:
                return (
                    CurveRegime.BULL_FLATTENER,
                    f"10Y yield dropped {dy10:.1f} bps, outpacing 2Y ({dy2:.1f} bps). "
                    f"2s10s slope flattened by {slope_delta:.1f} bps (Long-term growth downgrade).",
                )
            else:
                return (
                    CurveRegime.PARALLEL_SHIFT_DOWN,
                    f"Uniform downward shift across curve (2Y: {dy2:+.1f} bps, 10Y: {dy10:+.1f} bps).",
                )

        else:
            if slope_delta < 0:
                return CurveRegime.BEAR_FLATTENER, f"Curve flattened by {slope_delta:.1f} bps."
            return CurveRegime.BULL_STEEPENER, f"Curve steepened by {slope_delta:+.1f} bps."

    def _compute_intraday_slope_series(
        self, shy_window: MarketWindow, ief_window: MarketWindow, d2: float, d10: float
    ) -> pd.Series:
        """Calculate the minute-by-minute 2s10s slope change from T0 in basis points."""
        shy_norm = shy_window.normalized_prices
        ief_norm = ief_window.normalized_prices

        # Price percentage moves from T0
        shy_pct = (shy_norm - 100.0) / 100.0
        ief_pct = (ief_norm - 100.0) / 100.0

        # Yield moves
        dy2 = -(shy_pct / d2) * 10000.0
        dy10 = -(ief_pct / d10) * 10000.0

        slope_bps = dy10 - dy2
        return slope_bps
