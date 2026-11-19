"""
QuantNexus-Macro: Economic Release Calendar & Standardized Surprise Engine
Calculates consensus deviations, forecast standard errors, and standardized surprise Z-scores.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from quantnexus_macro.config import (
    MacroEventType,
    HISTORICAL_FORECAST_SIGMA,
)


@dataclass
class MacroRelease:
    event_id: str
    event_type: MacroEventType
    name: str
    timestamp: pd.Timestamp  # UTC timestamp of the release
    actual: float
    consensus: float
    prior: float
    unit: str
    description: str = ""

    @property
    def surprise(self) -> float:
        """Raw surprise: Actual - Consensus."""
        return self.actual - self.consensus

    @property
    def surprise_prior(self) -> float:
        """Revision surprise: Actual - Prior."""
        return self.actual - self.prior

    @property
    def sigma(self) -> float:
        """Historical forecast error standard deviation for this event type."""
        return HISTORICAL_FORECAST_SIGMA.get(self.event_type, 1.0)

    @property
    def z_score(self) -> float:
        """
        Standardized macroeconomic surprise metric:
        Z = (Actual - Consensus) / sigma_forecast_error
        """
        sig = self.sigma
        if sig == 0.0 or np.isnan(sig):
            return 0.0
        return self.surprise / sig

    @property
    def is_significant(self) -> bool:
        """Flag if surprise exceeds 1.0 standard deviation."""
        return abs(self.z_score) >= 1.0

    @property
    def is_extreme(self) -> bool:
        """Flag if surprise exceeds 2.0 standard deviations."""
        return abs(self.z_score) >= 2.0

    @property
    def policy_bias(self) -> str:
        """Categorize immediate monetary policy implication."""
        z = self.z_score
        if self.event_type in (MacroEventType.CPI, MacroEventType.CORE_CPI, MacroEventType.PPI):
            if z >= 1.0:
                return "Hawkish (Inflation Acceleration)"
            elif z <= -1.0:
                return "Dovish (Disinflation Confirmation)"
            return "Neutral / Inline"
        elif self.event_type == MacroEventType.NFP:
            if z >= 1.0:
                return "Hawkish (Tight Labor Market / Wage Pressure)"
            elif z <= -1.0:
                return "Dovish (Labor Market Softening)"
            return "Neutral / Inline"
        elif self.event_type == MacroEventType.FOMC_RATE_DECISION:
            if z > 0:
                return "Hawkish Rate Surprise"
            elif z < 0:
                return "Dovish Rate Surprise"
            return "Neutral / As Expected"
        return "Macro Impulse"


class EconomicCalendar:
    """
    Institutional repository for macroeconomic events, consensus figures,
    and standardized surprise calculation.
    """

    def __init__(self):
        self._releases: Dict[str, MacroRelease] = {}
        self._populate_historical_benchmark_releases()

    def add_release(self, release: MacroRelease) -> None:
        """Register a new macroeconomic release."""
        self._releases[release.event_id] = release

    def get_release(self, event_id: str) -> Optional[MacroRelease]:
        """Fetch release by unique identifier."""
        return self._releases.get(event_id)

    def list_releases(
        self,
        event_type: Optional[MacroEventType] = None,
        min_abs_z: float = 0.0,
    ) -> List[MacroRelease]:
        """List releases optionally filtered by type and minimum Z-score magnitude."""
        results = []
        for r in self._releases.values():
            if event_type and r.event_type != event_type:
                continue
            if abs(r.z_score) < min_abs_z:
                continue
            results.append(r)
        return sorted(results, key=lambda x: x.timestamp)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert all calendar events to a structured DataFrame."""
        records = []
        for r in self._releases.values():
            records.append({
                "event_id": r.event_id,
                "event_type": r.event_type.value,
                "name": r.name,
                "timestamp_utc": r.timestamp,
                "actual": r.actual,
                "consensus": r.consensus,
                "prior": r.prior,
                "surprise": r.surprise,
                "z_score": round(r.z_score, 2),
                "policy_bias": r.policy_bias,
                "unit": r.unit,
            })
        if not records:
            return pd.DataFrame()
        df = pd.DataFrame(records).sort_values("timestamp_utc").reset_index(drop=True)
        return df

    def _populate_historical_benchmark_releases(self):
        """
        Curated benchmark historical releases spanning critical market shocks.
        These serve as anchor historical cohorts for confidence cones and backtesting.
        """
        benchmarks = [
            # High-Impact CPI releases
            MacroRelease(
                event_id="CPI_2024_01_11",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (Dec 2023)",
                timestamp=pd.Timestamp("2024-01-11 13:30:00", tz="UTC"),
                actual=3.4,
                consensus=3.2,
                prior=3.1,
                unit="%",
                description="Hot CPI release triggering rate repricing higher",
            ),
            MacroRelease(
                event_id="CPI_2024_02_13",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (Jan 2024)",
                timestamp=pd.Timestamp("2024-02-13 13:30:00", tz="UTC"),
                actual=3.1,
                consensus=2.9,
                prior=3.4,
                unit="%",
                description="Persistent inflation shock; S&P sold off >1.5%",
            ),
            MacroRelease(
                event_id="CPI_2024_06_12",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (May 2024)",
                timestamp=pd.Timestamp("2024-06-12 12:30:00", tz="UTC"),
                actual=3.3,
                consensus=3.4,
                prior=3.4,
                unit="%",
                description="Dovish disinflation print spark rally across risk assets",
            ),
            MacroRelease(
                event_id="CPI_2024_07_11",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (Jun 2024)",
                timestamp=pd.Timestamp("2024-07-11 12:30:00", tz="UTC"),
                actual=3.0,
                consensus=3.1,
                prior=3.3,
                unit="%",
                description="Sub-3% headline print; massive rotation from tech to small caps",
            ),
            MacroRelease(
                event_id="CPI_2024_11_13",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (Oct 2024)",
                timestamp=pd.Timestamp("2024-11-13 13:30:00", tz="UTC"),
                actual=2.6,
                consensus=2.6,
                prior=2.4,
                unit="%",
                description="Exact inline release with stable cross-asset volatility",
            ),
            MacroRelease(
                event_id="CPI_2025_01_15",
                event_type=MacroEventType.CPI,
                name="US CPI Headline YoY (Dec 2024)",
                timestamp=pd.Timestamp("2025-01-15 13:30:00", tz="UTC"),
                actual=2.9,
                consensus=2.7,
                prior=2.7,
                unit="%",
                description="Inflation bounce; short-end rates repriced hawkishly",
            ),
            # Key NFP Releases
            MacroRelease(
                event_id="NFP_2024_08_02",
                event_type=MacroEventType.NFP,
                name="US Non-Farm Payrolls (Jul 2024)",
                timestamp=pd.Timestamp("2024-08-02 12:30:00", tz="UTC"),
                actual=114.0,
                consensus=175.0,
                prior=179.0,
                unit="k",
                description="Major growth scare, triggered Yen carry-trade unwind & Sahm rule discussion",
            ),
            MacroRelease(
                event_id="NFP_2024_10_04",
                event_type=MacroEventType.NFP,
                name="US Non-Farm Payrolls (Sep 2024)",
                timestamp=pd.Timestamp("2024-10-04 12:30:00", tz="UTC"),
                actual=254.0,
                consensus=140.0,
                prior=159.0,
                unit="k",
                description="Massive upside beat (+1.52 sigma); crushed November 50bps Fed cut bets",
            ),
            # Key FOMC Decisions
            MacroRelease(
                event_id="FOMC_2024_09_18",
                event_type=MacroEventType.FOMC_RATE_DECISION,
                name="FOMC 50bps Rate Cut Decision",
                timestamp=pd.Timestamp("2024-09-18 18:00:00", tz="UTC"),
                actual=4.875,
                consensus=5.000,
                prior=5.375,
                unit="%",
                description="Fed delivered jumbo 50bps cut; steepener curve rally",
            ),
            MacroRelease(
                event_id="FOMC_2024_11_07",
                event_type=MacroEventType.FOMC_RATE_DECISION,
                name="FOMC 25bps Rate Cut Decision",
                timestamp=pd.Timestamp("2024-11-07 19:00:00", tz="UTC"),
                actual=4.625,
                consensus=4.625,
                prior=4.875,
                unit="%",
                description="Expected 25 bps cut with neutral press conference tone",
            ),
        ]

        for b in benchmarks:
            self.add_release(b)
