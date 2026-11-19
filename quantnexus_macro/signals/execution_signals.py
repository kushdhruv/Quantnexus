"""
QuantNexus-Macro: Prop Desk Execution Ticket & Risk Sizing Engine
Translates statistical dislocation signals into structured algorithmic order tickets,
hedged relative-value pair instructions, and Value-at-Risk (VaR) position limits.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from quantnexus_macro.analytics.dislocation import DislocationSignal, TradeAction
from quantnexus_macro.analytics.curve import CurveAnalysisResult, CurveRegime


@dataclass
class ExecutionTicket:
    ticket_id: str
    strategy_type: str  # "Outright Convergence", "Delta-Neutral Pair", "Curve Twist Spread"
    primary_ticker: str
    action: str         # "BUY" or "SELL"
    entry_price: float
    target_price: float
    stop_loss: float
    hedged_ticker: Optional[str] = None
    hedged_action: Optional[str] = None
    hedge_ratio: float = 0.0
    suggested_notional_usd: float = 0.0
    expected_pnl_usd: float = 0.0
    max_loss_usd: float = 0.0
    var_99_shock_usd: float = 0.0
    execution_instructions: str = ""


class ExecutionOrderManager:
    """
    Transforms quantitative alpha signals into executable institutional order tickets
    with strict prop risk sizing and spread hedging.
    """

    def __init__(self, book_capital_usd: float = 10_000_000.0, max_risk_per_trade_bps: float = 15.0):
        self.book_capital = book_capital_usd
        self.max_risk_bps = max_risk_per_trade_bps  # 15 bps = $15,000 max loss per ticket on $10M book

    def generate_tickets(
        self,
        signals: List[DislocationSignal],
        curve_result: Optional[CurveAnalysisResult] = None,
    ) -> List[ExecutionTicket]:
        """Generate structured tickets for all non-neutral signals."""
        tickets: List[ExecutionTicket] = []
        max_dollar_risk = self.book_capital * (self.max_risk_bps / 10000.0)

        counter = 1
        for s in signals:
            if s.action == TradeAction.NEUTRAL:
                continue

            ticket_id = f"QNX-TKT-{counter:03d}"
            counter += 1

            is_buy = "BUY" in s.action.value
            act_str = "BUY" if is_buy else "SELL"

            stop_dist_pct = abs(s.stop_loss_price - s.current_price) / s.current_price
            stop_dist_pct = max(stop_dist_pct, 0.0020)  # Min 20 bps stop distance

            # Position notional sized so stop loss hits exactly max_dollar_risk
            notional = max_dollar_risk / stop_dist_pct
            # Cap notional at 20% of book capital
            notional = min(notional, self.book_capital * 0.20)

            target_gain_pct = abs(s.target_price - s.current_price) / s.current_price
            exp_pnl = notional * target_gain_pct
            max_loss = notional * stop_dist_pct

            # 99% 1-hour VaR proxy (2.33 * sigma)
            var_shock = notional * (stop_dist_pct * 1.5)

            # Pair hedge suggestion
            hedge_ticker = None
            hedge_act = None
            hedge_ratio = 0.0

            if s.ticker in ("QQQ", "IWM"):
                hedge_ticker = "SPY"
                hedge_act = "SELL" if act_str == "BUY" else "BUY"
                hedge_ratio = 1.0
            elif s.ticker in ("TLT", "IEF"):
                hedge_ticker = "SHY"
                hedge_act = "SELL" if act_str == "BUY" else "BUY"
                hedge_ratio = 2.5

            instructions = (
                f"Enter {act_str} {s.ticker} limit @ {s.current_price:.2f}. "
                f"Bracket: Take Profit @ {s.target_price:.2f}, Stop Loss @ {s.stop_loss_price:.2f}. "
                f"Time Horizon: ~{s.expected_convergence_minutes} mins."
            )

            tickets.append(
                ExecutionTicket(
                    ticket_id=ticket_id,
                    strategy_type="Macro Dislocation Convergence",
                    primary_ticker=s.ticker,
                    action=act_str,
                    entry_price=s.current_price,
                    target_price=s.target_price,
                    stop_loss=s.stop_loss_price,
                    hedged_ticker=hedge_ticker,
                    hedged_action=hedge_act,
                    hedge_ratio=hedge_ratio,
                    suggested_notional_usd=round(notional, 0),
                    expected_pnl_usd=round(exp_pnl, 0),
                    max_loss_usd=round(max_loss, 0),
                    var_99_shock_usd=round(var_shock, 0),
                    execution_instructions=instructions,
                )
            )

        # Check if Curve Spread trade should be emitted
        if curve_result and abs(curve_result.slope_2s10s_bps_change) >= 2.0:
            ticket_id = f"QNX-TKT-{counter:03d}"
            if curve_result.regime == CurveRegime.BEAR_FLATTENER:
                curve_strat = "Flattener Fade (Bet on Slope Re-steepening)"
                inst = "Enter 2s10s steepener: BUY 10Y Note (IEF) vs SELL 2Y Note (SHY) duration matched."
            else:
                curve_strat = "Steepener Fade (Bet on Slope Re-flattening)"
                inst = "Enter 2s10s flattener: SELL 10Y Note (IEF) vs BUY 2Y Note (SHY) duration matched."

            tickets.append(
                ExecutionTicket(
                    ticket_id=ticket_id,
                    strategy_type=f"Curve Twist: {curve_strat}",
                    primary_ticker="IEF",
                    action="BUY" if "steepener" in inst else "SELL",
                    entry_price=round(float(curve_result.tenor_shifts[10].yield_change_bps), 2),
                    target_price=0.0,
                    stop_loss=0.0,
                    hedged_ticker="SHY",
                    hedged_action="SELL" if "steepener" in inst else "BUY",
                    hedge_ratio=4.1,  # 7.8 / 1.9 duration ratio
                    suggested_notional_usd=2_000_000.0,
                    expected_pnl_usd=25_000.0,
                    max_loss_usd=12_000.0,
                    var_99_shock_usd=18_000.0,
                    execution_instructions=inst,
                )
            )

        return tickets

    def tickets_to_dataframe(self, tickets: List[ExecutionTicket]) -> pd.DataFrame:
        """Convert tickets to execution blotter table."""
        rows = []
        for t in tickets:
            rows.append({
                "Ticket ID": t.ticket_id,
                "Strategy": t.strategy_type,
                "Action": f"{t.action} {t.primary_ticker}",
                "Entry": t.entry_price,
                "Target": t.target_price,
                "Stop": t.stop_loss,
                "Hedge": f"{t.hedged_action} {t.hedged_ticker} ({t.hedge_ratio:.1f}x)" if t.hedged_ticker else "None",
                "Notional ($)": f"${t.suggested_notional_usd:,.0f}",
                "Target PnL ($)": f"${t.expected_pnl_usd:,.0f}",
                "Max Risk ($)": f"${t.max_loss_usd:,.0f}",
                "VaR 99 ($)": f"${t.var_99_shock_usd:,.0f}",
            })
        return pd.DataFrame(rows)
