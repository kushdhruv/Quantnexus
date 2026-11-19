"""
QuantNexus-Macro: Master Orchestration Engine
Unifies market ingestion, econometric event study, yield curve decomposition,
lead-lag cascade, regime detection, MTDE dislocation alpha, and reporting.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Union
import pandas as pd

from quantnexus_macro.config import (
    DEFAULT_ASSET_UNIVERSE,
    DEFAULT_PRE_EVENT_MINUTES,
    DEFAULT_POST_EVENT_MINUTES,
    MacroEventType,
)
from quantnexus_macro.core.calendar import EconomicCalendar, MacroRelease
from quantnexus_macro.core.data_feed import DataFeedManager, MarketWindow
from quantnexus_macro.core.event_study import EventStudyEngine, EventStudyMetrics
from quantnexus_macro.analytics.curve import YieldCurveAnalyzer, CurveAnalysisResult
from quantnexus_macro.analytics.lead_lag import LeadLagAnalyzer, AbsorptionVelocityResult, LeadLagPairResult
from quantnexus_macro.analytics.regime import MacroRegimeClassifier, RegimeClassificationResult
from quantnexus_macro.analytics.dislocation import MacroDislocationEngine, DislocationMatrixResult
from quantnexus_macro.analytics.cohort_backtest import HistoricalCohortAnalyzer, CohortBenchmarkResult
from quantnexus_macro.signals.execution_signals import ExecutionOrderManager, ExecutionTicket
from quantnexus_macro.visualization.report_generator import MacroReportGenerator


@dataclass
class MacroAnalysisPackage:
    event: MacroRelease
    windows: Dict[str, MarketWindow]
    event_study_metrics: Dict[str, EventStudyMetrics]
    curve_result: Optional[CurveAnalysisResult]
    regime_result: RegimeClassificationResult
    dislocation_result: DislocationMatrixResult
    cohort_benchmark: Optional[CohortBenchmarkResult]
    absorption_velocities: Dict[str, AbsorptionVelocityResult]
    execution_tickets: List[ExecutionTicket]
    report_html_path: Optional[str] = None


class QuantNexusEngine:
    """
    Institutional End-to-End Quantitative Macro Transmission Platform.
    """

    def __init__(
        self,
        cache_dir: str = ".quantnexus_cache",
        reports_dir: str = "reports",
        book_capital_usd: float = 10_000_000.0,
    ):
        self.calendar = EconomicCalendar()
        self.data_feed = DataFeedManager(cache_dir=cache_dir)
        self.event_study = EventStudyEngine()
        self.curve_analyzer = YieldCurveAnalyzer()
        self.lead_lag = LeadLagAnalyzer()
        self.regime_classifier = MacroRegimeClassifier()
        self.dislocation_engine = MacroDislocationEngine()
        self.cohort_analyzer = HistoricalCohortAnalyzer(calendar=self.calendar, data_feed=self.data_feed)
        self.order_manager = ExecutionOrderManager(book_capital_usd=book_capital_usd)
        self.report_generator = MacroReportGenerator(output_dir=reports_dir)

    def analyze_event(
        self,
        event_or_id: Union[str, MacroRelease],
        tickers: Optional[List[str]] = None,
        pre_event_minutes: int = DEFAULT_PRE_EVENT_MINUTES,
        post_event_minutes: int = DEFAULT_POST_EVENT_MINUTES,
        benchmark_cohort_ticker: str = "SPY",
        generate_report: bool = True,
        force_simulation: bool = False,
    ) -> MacroAnalysisPackage:
        """
        Execute comprehensive institutional end-to-end event analysis.
        """
        # Resolve MacroRelease
        if isinstance(event_or_id, str):
            release = self.calendar.get_release(event_or_id)
            if release is None:
                raise ValueError(f"Event ID '{event_or_id}' not found in calendar.")
        else:
            release = event_or_id

        if tickers is None:
            tickers = list(DEFAULT_ASSET_UNIVERSE.keys())

        # 1. Ingest Market Data
        windows = self.data_feed.fetch_cross_asset_window(
            event=release,
            tickers=tickers,
            pre_event_minutes=pre_event_minutes,
            post_event_minutes=post_event_minutes,
            force_simulation=force_simulation,
        )

        # 2. Econometric Event Study
        study_metrics = self.event_study.analyze_cross_asset_universe(windows)

        # 3. Yield Curve Decomposition
        curve_result = self.curve_analyzer.analyze_curve_response(windows)

        # 4. Lead-Lag & Absorption Speeds
        velocities = self.lead_lag.compute_absorption_velocities(windows)

        # 5. Macro Regime Classification
        regime_result = self.regime_classifier.classify_market_response(windows)

        # 6. [GAME CHANGER] Macro Transmission Dislocation Engine (MTDE)
        dislocation_result = self.dislocation_engine.analyze_dislocations(
            windows=windows,
            event=release,
        )

        # 7. Historical Cohort Confidence Cone
        cohort_benchmark = None
        if benchmark_cohort_ticker in windows:
            cohort_benchmark = self.cohort_analyzer.benchmark_event_against_cone(
                current_window=windows[benchmark_cohort_ticker],
                current_event=release,
            )

        # 8. Execution Tickets & Risk Management
        tickets = self.order_manager.generate_tickets(
            signals=dislocation_result.signals,
            curve_result=curve_result,
        )

        # 9. Optional HTML Report Generation
        report_path = None
        if generate_report:
            report_path = self.report_generator.generate_html_report(
                event=release,
                windows=windows,
                study_metrics=study_metrics,
                curve_result=curve_result,
                regime_result=regime_result,
                dislocation_result=dislocation_result,
                cohort_result=cohort_benchmark,
                lead_lag_velocities=velocities,
                tickets=tickets,
            )

        return MacroAnalysisPackage(
            event=release,
            windows=windows,
            event_study_metrics=study_metrics,
            curve_result=curve_result,
            regime_result=regime_result,
            dislocation_result=dislocation_result,
            cohort_benchmark=cohort_benchmark,
            absorption_velocities=velocities,
            execution_tickets=tickets,
            report_html_path=report_path,
        )


def build_macro_tracker(
    event_name: str = "CPI Impact Study",
    event_time_str: str = "2024-02-13 13:30:00",
    tickers: Optional[List[str]] = None,
    actual: Optional[float] = None,
    consensus: Optional[float] = None,
    pre_minutes: int = 30,
    post_minutes: int = 120,
    show: bool = True,
    force_simulation: bool = False,
):
    """
    1-to-1 Drop-in Institutional Upgrade for the original build_macro_tracker interface.
    Fully compatible with https://github.com/amit-singh-99/cross-asset-macro-tracker,
    while running the full institutional-grade multi-asset transmission and MTDE dislocation engine.
    """
    engine = QuantNexusEngine()

    # Parse timestamp
    ts = pd.to_datetime(event_time_str)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    # Determine event type from event_name
    name_upper = event_name.upper()
    if "CPI" in name_upper:
        ev_type = MacroEventType.CPI
        default_actual = 3.1 if actual is None else actual
        default_cons = 2.9 if consensus is None else consensus
        unit = "%"
    elif "NFP" in name_upper or "PAYROLL" in name_upper:
        ev_type = MacroEventType.NFP
        default_actual = 254.0 if actual is None else actual
        default_cons = 140.0 if consensus is None else consensus
        unit = "k"
    elif "FOMC" in name_upper or "RATE" in name_upper:
        ev_type = MacroEventType.FOMC_RATE_DECISION
        default_actual = 4.875 if actual is None else actual
        default_cons = 5.000 if consensus is None else consensus
        unit = "%"
    else:
        ev_type = MacroEventType.CPI
        default_actual = 3.0 if actual is None else actual
        default_cons = 3.0 if consensus is None else consensus
        unit = "%"

    # Default tickers (covering all original assets SPY, UUP, TLT, GLD + rates)
    if tickers is None:
        tickers = ["SPY", "UUP", "TLT", "GLD", "SHY", "IEF"]

    release = MacroRelease(
        event_id=f"CUSTOM_{ts.strftime('%Y%m%d_%H%M%S')}",
        event_type=ev_type,
        name=event_name,
        timestamp=ts,
        actual=default_actual,
        consensus=default_cons,
        prior=default_cons,
        unit=unit,
        description=f"Ad-hoc analysis for {event_name}",
    )

    pkg = engine.analyze_event(
        event_or_id=release,
        tickers=tickers,
        pre_event_minutes=pre_minutes,
        post_event_minutes=post_minutes,
        force_simulation=force_simulation,
        generate_report=True,
    )

    # Master figure
    fig = engine.report_generator.dashboard.create_master_figure(
        event=release,
        windows=pkg.windows,
        curve_result=pkg.curve_result,
        dislocation_result=pkg.dislocation_result,
        cohort_result=pkg.cohort_benchmark,
    )

    if show:
        try:
            fig.show()
        except Exception:
            pass

    return fig
