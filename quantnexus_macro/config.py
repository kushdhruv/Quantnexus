"""
QuantNexus-Macro: Configuration & Asset Universe Definitions
Institutional-grade cross-asset universe and macroeconomic indicator specifications.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class AssetClass(str, Enum):
    EQUITIES = "Equities"
    RATES = "Fixed Income / Rates"
    FX = "Foreign Exchange"
    COMMODITIES = "Commodities"
    CREDIT = "Credit"
    VOLATILITY = "Volatility"


class MacroEventType(str, Enum):
    CPI = "Consumer Price Index (CPI)"
    CORE_CPI = "Core CPI MoM"
    NFP = "Non-Farm Payrolls (NFP)"
    FOMC_RATE_DECISION = "FOMC Rate Decision"
    PPI = "Producer Price Index (PPI)"
    RETAIL_SALES = "Retail Sales MoM"
    ISM_MANUFACTURING = "ISM Manufacturing PMI"
    GDP_ANNUALIZED = "GDP QoQ Annualized"


class MacroRegime(str, Enum):
    HAWKISH_INFLATION = "Hawkish Inflation Shock (Rates Up, USD Up, Stocks Down)"
    DOVISH_DISINFLATION = "Dovish Disinflation Pivot (Rates Down, USD Down, Stocks Up)"
    GROWTH_RECESSION = "Growth / Recession Scare (Rates Down, Flight-to-Quality, Stocks Down)"
    REFLATION_RISK_ON = "Reflationary Risk-On (Rates Up, USD Down, Stocks Up, Commodities Up)"
    LIQUIDITY_CONTRACTION = "Liquidity Shock / Contraction (Everything Down, USD/VIX Up)"
    NEUTRAL_NOISE = "Indeterminate / Sub-Threshold Noise"


@dataclass(frozen=True)
class AssetMeta:
    ticker: str
    name: str
    asset_class: AssetClass
    benchmark_yield_tenor: Optional[int] = None  # in years (e.g., 2, 5, 10, 20)
    duration_proxy: float = 0.0  # approximate modified duration
    description: str = ""
    currency: str = "USD"


# Institutional Cross-Asset Universe configured for Prop Desks (Rates, FX, Indices, Commodities)
DEFAULT_ASSET_UNIVERSE: Dict[str, AssetMeta] = {
    # Equities
    "SPY": AssetMeta(
        ticker="SPY",
        name="S&P 500 Index (SPY)",
        asset_class=AssetClass.EQUITIES,
        duration_proxy=0.0,
        description="US Large Cap Benchmark Equity ETF",
    ),
    "QQQ": AssetMeta(
        ticker="QQQ",
        name="Nasdaq 100 (QQQ)",
        asset_class=AssetClass.EQUITIES,
        duration_proxy=0.0,
        description="High Duration Tech & Growth Benchmark ETF",
    ),
    "IWM": AssetMeta(
        ticker="IWM",
        name="Russell 2000 (IWM)",
        asset_class=AssetClass.EQUITIES,
        duration_proxy=0.0,
        description="US Small Cap & Credit Sensitive Equities",
    ),
    # Fixed Income / Yield Curve Tenors
    "SHY": AssetMeta(
        ticker="SHY",
        name="US Treasury 1-3Y (2Y Proxy)",
        asset_class=AssetClass.RATES,
        benchmark_yield_tenor=2,
        duration_proxy=1.9,
        description="Short-Term Rate Policy Sensitive Tenor",
    ),
    "IEI": AssetMeta(
        ticker="IEI",
        name="US Treasury 3-7Y (5Y Proxy)",
        asset_class=AssetClass.RATES,
        benchmark_yield_tenor=5,
        duration_proxy=4.5,
        description="Intermediate Belly Tenor",
    ),
    "IEF": AssetMeta(
        ticker="IEF",
        name="US Treasury 7-10Y (10Y Proxy)",
        asset_class=AssetClass.RATES,
        benchmark_yield_tenor=10,
        duration_proxy=7.8,
        description="Benchmark Global Benchmark Yield Tenor",
    ),
    "TLT": AssetMeta(
        ticker="TLT",
        name="US Treasury 20Y+ (20Y/30Y Proxy)",
        asset_class=AssetClass.RATES,
        benchmark_yield_tenor=20,
        duration_proxy=16.5,
        description="Ultra-Long Duration & Term Premium Tenor",
    ),
    # Foreign Exchange (Dollar Index & Key G10 Currencies)
    "UUP": AssetMeta(
        ticker="UUP",
        name="US Dollar Index (DXY Bullish)",
        asset_class=AssetClass.FX,
        duration_proxy=0.0,
        description="Trade-Weighted US Dollar Index ETF",
    ),
    "FXE": AssetMeta(
        ticker="FXE",
        name="Euro FX (EUR/USD)",
        asset_class=AssetClass.FX,
        duration_proxy=0.0,
        description="Euro Currency Trust",
    ),
    "FXY": AssetMeta(
        ticker="FXY",
        name="Japanese Yen (USD/JPY Inverted)",
        asset_class=AssetClass.FX,
        duration_proxy=0.0,
        description="Japanese Yen Currency Trust",
    ),
    # Commodities
    "GLD": AssetMeta(
        ticker="GLD",
        name="Gold Bullion (GLD)",
        asset_class=AssetClass.COMMODITIES,
        duration_proxy=0.0,
        description="Monetary Gold / Real Rate Hedge",
    ),
    "USO": AssetMeta(
        ticker="USO",
        name="WTI Crude Oil (USO)",
        asset_class=AssetClass.COMMODITIES,
        duration_proxy=0.0,
        description="Cyclical Energy / Supply-Demand Shock Proxy",
    ),
    # Credit & Risk Appetite
    "HYG": AssetMeta(
        ticker="HYG",
        name="High Yield Corporate Bond (HYG)",
        asset_class=AssetClass.CREDIT,
        duration_proxy=3.8,
        description="Credit Spread & Default Risk Sentiment",
    ),
}

# Empirical standard deviations of forecast errors for Z-score normalization
# Based on historical Bloomberg / consensus forecast survey errors over rolling 5Y
HISTORICAL_FORECAST_SIGMA: Dict[MacroEventType, float] = {
    MacroEventType.CPI: 0.15,  # 0.15% YoY surprise standard deviation
    MacroEventType.CORE_CPI: 0.08,  # 0.08% MoM surprise standard deviation
    MacroEventType.NFP: 75.0,  # 75k payrolls forecast standard deviation
    MacroEventType.FOMC_RATE_DECISION: 12.5,  # 12.5 bps policy surprise
    MacroEventType.PPI: 0.20,  # 0.20% MoM
    MacroEventType.RETAIL_SALES: 0.35,  # 0.35% MoM
    MacroEventType.ISM_MANUFACTURING: 1.2,  # 1.2 index points
    MacroEventType.GDP_ANNUALIZED: 0.60,  # 0.60% annualized
}

# Default theoretical macro beta priors (sign and relative scale of 1-minute to 60-minute impact per +1 sigma surprise)
# Hawkish surprise (e.g. Higher Inflation / Higher Rates):
HAWKING_SURPRISE_BETAS: Dict[str, float] = {
    "SPY": -0.65,   # Equities decline due to higher discount rates
    "QQQ": -0.95,   # Tech suffers higher duration penalty
    "IWM": -0.80,   # Small caps hit by credit tightness
    "SHY": -0.40,   # 2Y bond prices drop (yields spike)
    "IEI": -0.75,   # 5Y bond prices drop
    "IEF": -0.90,   # 10Y bond prices drop
    "TLT": -1.35,   # 20Y bond prices drop heavily
    "UUP": +0.55,   # Dollar index rallies on yield advantage
    "FXE": -0.50,   # EUR drops vs USD
    "FXY": -0.70,   # JPY drops on widening yield differential
    "GLD": -0.45,   # Gold drops on real rate increase
    "USO": +0.20,   # Energy may drift or absorb demand signals
    "HYG": -0.60,   # Credit spreads widen
}

# Default Event Study Window Parameters (minutes relative to T0)
DEFAULT_PRE_EVENT_MINUTES: int = 30
DEFAULT_POST_EVENT_MINUTES: int = 120
DEFAULT_SAMPLING_FREQUENCY: str = "1m"
