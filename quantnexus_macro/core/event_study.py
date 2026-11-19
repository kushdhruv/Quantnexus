"""
QuantNexus-Macro: Econometric Event Study Engine
Implements Campbell-Lo-MacKinlay event study methodology with exact T0 normalization,
Cumulative Abnormal Returns (CAR), volatility estimation, and standardized test statistics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from scipy import stats

from quantnexus_macro.core.data_feed import MarketWindow


@dataclass
class EventStudyMetrics:
    ticker: str
    name: str
    t0_price: float
    final_price: float
    post_event_return_pct: float  # Cumulative return from T0 to window end (%)
    car_pct: float                # Cumulative Abnormal Return (%)
    max_favorable_excursion_pct: float  # MFE (%)
    max_adverse_excursion_pct: float    # MAE (%)
    realized_vol_pre_event_ann: float   # Pre-event annualized volatility
    realized_vol_post_event_ann: float  # Post-event annualized volatility
    vol_expansion_ratio: float          # Post-vol / Pre-vol
    t_statistic: float                  # Standardized CAR test statistic
    p_value: float                      # Two-tailed significance p-value
    is_statistically_significant: bool  # p < 0.05


class EventStudyEngine:
    """
    Econometric Event Study Analyzer.
    Provides rigorous statistical testing of price impact across event horizons.
    """

    def __init__(self, pre_event_window: int = 30, post_event_window: int = 120):
        self.pre_event_window = pre_event_window
        self.post_event_window = post_event_window

    def analyze_window(self, window: MarketWindow) -> EventStudyMetrics:
        """Compute econometric event study metrics for a single market window."""
        rel_min = window.relative_minutes
        prices = window.prices.values
        norm = window.normalized_prices.values

        # Indices for T0 and pre/post windows
        t0_idx = np.where(rel_min == 0)[0]
        if len(t0_idx) == 0:
            # Fallback to middle if exact 0 not found
            t0_idx = len(rel_min) // 2
        else:
            t0_idx = t0_idx[0]

        t0_price = prices[t0_idx]
        final_price = prices[-1]

        # Post-event returns from T0 onwards
        post_prices = prices[t0_idx:]
        post_returns = np.diff(post_prices) / post_prices[:-1]
        cumulative_return_pct = ((final_price / t0_price) - 1.0) * 100.0

        # Pre-event estimation window returns
        pre_prices = prices[:t0_idx + 1]
        pre_returns = np.diff(pre_prices) / pre_prices[:-1] if len(pre_prices) > 1 else np.array([0.0])

        # Pre-event baseline expected drift
        expected_drift = np.mean(pre_returns) if len(pre_returns) > 0 else 0.0
        pre_vol_minute = np.std(pre_returns, ddof=1) if len(pre_returns) > 2 else 0.0005
        post_vol_minute = np.std(post_returns, ddof=1) if len(post_returns) > 2 else 0.0005

        # Annualized vol factors (252 trading days * 390 minutes/day)
        ann_factor = np.sqrt(252.0 * 390.0)
        pre_vol_ann = pre_vol_minute * ann_factor
        post_vol_ann = post_vol_minute * ann_factor
        vol_expansion = post_vol_ann / max(pre_vol_ann, 1e-6)

        # Cumulative Abnormal Return (CAR)
        # AR_t = R_t - expected_drift
        abnormal_returns = post_returns - expected_drift
        car = np.sum(abnormal_returns)
        car_pct = car * 100.0

        # Max Favorable / Adverse Excursion relative to T0
        pct_trajectory = ((post_prices / t0_price) - 1.0) * 100.0
        mfe_pct = float(np.max(pct_trajectory))
        mae_pct = float(np.min(pct_trajectory))

        # Standardized Test Statistic (Boehmer-Musumeci-Poulsen / Patell proxy)
        # Var(CAR) = T * sigma^2
        horizon_steps = len(abnormal_returns)
        variance_car = horizon_steps * (pre_vol_minute ** 2)
        std_error_car = np.sqrt(max(variance_car, 1e-10))
        t_stat = car / std_error_car if std_error_car > 0 else 0.0
        p_val = float(2.0 * (1.0 - stats.norm.cdf(abs(t_stat))))

        return EventStudyMetrics(
            ticker=window.ticker,
            name=window.meta.name,
            t0_price=round(float(t0_price), 4),
            final_price=round(float(final_price), 4),
            post_event_return_pct=round(cumulative_return_pct, 2),
            car_pct=round(car_pct, 2),
            max_favorable_excursion_pct=round(mfe_pct, 2),
            max_adverse_excursion_pct=round(mae_pct, 2),
            realized_vol_pre_event_ann=round(float(pre_vol_ann), 4),
            realized_vol_post_event_ann=round(float(post_vol_ann), 4),
            vol_expansion_ratio=round(float(vol_expansion), 2),
            t_statistic=round(float(t_stat), 2),
            p_value=round(p_val, 4),
            is_statistically_significant=bool(p_val < 0.05),
        )

    def analyze_cross_asset_universe(
        self, windows: Dict[str, MarketWindow]
    ) -> Dict[str, EventStudyMetrics]:
        """Run econometric event study across all assets in the universe."""
        metrics = {}
        for ticker, window in windows.items():
            metrics[ticker] = self.analyze_window(window)
        return metrics

    def metrics_to_dataframe(
        self, metrics: Dict[str, EventStudyMetrics]
    ) -> pd.DataFrame:
        """Convert metrics to a ranked tabular presentation."""
        rows = []
        for m in metrics.values():
            rows.append({
                "Ticker": m.ticker,
                "Name": m.name,
                "T0 Price": m.t0_price,
                "T+End Price": m.final_price,
                "Post Return (%)": m.post_event_return_pct,
                "CAR (%)": m.car_pct,
                "MFE (%)": m.max_favorable_excursion_pct,
                "MAE (%)": m.max_adverse_excursion_pct,
                "Pre Vol (Ann)": f"{m.realized_vol_pre_event_ann * 100:.1f}%",
                "Post Vol (Ann)": f"{m.realized_vol_post_event_ann * 100:.1f}%",
                "Vol Ratio": f"{m.vol_expansion_ratio:.2f}x",
                "t-stat": m.t_statistic,
                "p-val": m.p_value,
                "Sig (p<0.05)": "YES" if m.is_statistically_significant else "No",
            })
        return pd.DataFrame(rows).sort_values("Post Return (%)", ascending=False).reset_index(drop=True)
