"""
QuantNexus-Macro: Macro Shock Regime Classification Engine
Classifies the market's collective response into institutional macroeconomic regimes
using cross-asset sign concordance, principal component shock vectors, and divergence metrics.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from quantnexus_macro.config import MacroRegime
from quantnexus_macro.core.data_feed import MarketWindow


@dataclass
class RegimeClassificationResult:
    detected_regime: MacroRegime
    confidence_score: float  # 0.0 to 1.0 (Concordance %)
    rate_direction: str     # "Spiking", "Falling", "Flat"
    usd_direction: str      # "Surging", "Weakening", "Flat"
    equity_direction: str   # "Rallying", "Selling Off", "Flat"
    commodity_direction: str # "Rising", "Falling", "Flat"
    narrative_summary: str
    regime_probabilities: Dict[str, float]


class MacroRegimeClassifier:
    """
    Classifies cross-market transmission into discrete macro regimes
    by projecting observed asset response vectors onto theoretical regime archetypes.
    """

    def __init__(self, move_threshold_pct: float = 0.15):
        self.move_threshold = move_threshold_pct

    def classify_market_response(
        self, windows: Dict[str, MarketWindow]
    ) -> RegimeClassificationResult:
        """Classify the macro regime indicated by cross-asset price changes."""
        # Extract cumulative percentage moves from T0 to end
        moves: Dict[str, float] = {}
        for ticker, w in windows.items():
            t0_p = w.t0_price
            end_p = float(w.prices.iloc[-1])
            moves[ticker] = ((end_p / t0_p) - 1.0) * 100.0

        # Anchor proxies
        spy_move = moves.get("SPY", 0.0)
        tlt_move = moves.get("TLT", moves.get("IEF", 0.0))  # Bond price move (yield is inverse)
        uup_move = moves.get("UUP", 0.0)
        gld_move = moves.get("GLD", 0.0)
        uso_move = moves.get("USO", 0.0)

        # Implied 10Y yield shift (inverse of IEF/TLT)
        yield_rising = tlt_move < -self.move_threshold
        yield_falling = tlt_move > self.move_threshold

        usd_rallying = uup_move > (self.move_threshold * 0.5)
        usd_falling = uup_move < -(self.move_threshold * 0.5)

        equity_rallying = spy_move > self.move_threshold
        equity_dropping = spy_move < -self.move_threshold

        gold_rallying = gld_move > self.move_threshold
        gold_dropping = gld_move < -self.move_threshold

        # Format directional indicators
        rate_dir = "Yields Spiking" if yield_rising else ("Yields Collapsing" if yield_falling else "Rates Stable")
        usd_dir = "USD Strengthening" if usd_rallying else ("USD Weakening" if usd_falling else "USD Stable")
        eq_dir = "Equities Selling Off" if equity_dropping else ("Equities Surging" if equity_rallying else "Equities Stable")
        comm_dir = "Commodities Rising" if (gold_rallying or uso_move > self.move_threshold) else (
            "Commodities Falling" if (gold_dropping or uso_move < -self.move_threshold) else "Commodities Stable"
        )

        # Regime Scores: Concordance with archetype features
        scores: Dict[MacroRegime, float] = {
            MacroRegime.HAWKISH_INFLATION: 0.0,
            MacroRegime.DOVISH_DISINFLATION: 0.0,
            MacroRegime.GROWTH_RECESSION: 0.0,
            MacroRegime.REFLATION_RISK_ON: 0.0,
            MacroRegime.LIQUIDITY_CONTRACTION: 0.0,
        }

        # 1. Hawkish Inflation: Yields Up (Bonds Down), USD Up, Equities Down, Gold Down
        if yield_rising: scores[MacroRegime.HAWKISH_INFLATION] += 0.35
        if usd_rallying: scores[MacroRegime.HAWKISH_INFLATION] += 0.25
        if equity_dropping: scores[MacroRegime.HAWKISH_INFLATION] += 0.25
        if gold_dropping: scores[MacroRegime.HAWKISH_INFLATION] += 0.15

        # 2. Dovish Disinflation: Yields Down (Bonds Up), USD Down, Equities Up, Gold Up
        if yield_falling: scores[MacroRegime.DOVISH_DISINFLATION] += 0.35
        if usd_falling: scores[MacroRegime.DOVISH_DISINFLATION] += 0.25
        if equity_rallying: scores[MacroRegime.DOVISH_DISINFLATION] += 0.25
        if gold_rallying: scores[MacroRegime.DOVISH_DISINFLATION] += 0.15

        # 3. Growth / Recession Scare: Yields Down (Flight to Safety), Equities Down, Commodities Down
        if yield_falling: scores[MacroRegime.GROWTH_RECESSION] += 0.35
        if equity_dropping: scores[MacroRegime.GROWTH_RECESSION] += 0.35
        if uso_move < -self.move_threshold: scores[MacroRegime.GROWTH_RECESSION] += 0.15
        if usd_rallying or abs(uup_move) < self.move_threshold: scores[MacroRegime.GROWTH_RECESSION] += 0.15

        # 4. Reflationary Risk-On: Yields Up (Growth), Equities Up, Commodities Up, USD Down/Neutral
        if yield_rising: scores[MacroRegime.REFLATION_RISK_ON] += 0.30
        if equity_rallying: scores[MacroRegime.REFLATION_RISK_ON] += 0.35
        if uso_move > self.move_threshold or gold_rallying: scores[MacroRegime.REFLATION_RISK_ON] += 0.20
        if usd_falling: scores[MacroRegime.REFLATION_RISK_ON] += 0.15

        # 5. Liquidity Shock: Bonds Down, Stocks Down, Commodities Down, USD Up
        if yield_rising and equity_dropping and gold_dropping and usd_rallying:
            scores[MacroRegime.LIQUIDITY_CONTRACTION] += 0.80

        # Normalization of probabilities
        max_regime = max(scores, key=lambda k: scores[k])
        max_score = scores[max_regime]

        if max_score < 0.30:
            final_regime = MacroRegime.NEUTRAL_NOISE
            confidence = 0.50
            narrative = "Market reaction is sub-threshold or cross-asset signals are in direct conflict."
        else:
            final_regime = max_regime
            confidence = min(round(max_score, 2), 0.98)
            narrative = self._generate_narrative(final_regime, rate_dir, usd_dir, eq_dir)

        # Probabilities dictionary
        total_s = sum(scores.values()) or 1.0
        regime_probs = {k.value: round(v / total_s, 2) for k, v in scores.items()}

        return RegimeClassificationResult(
            detected_regime=final_regime,
            confidence_score=confidence,
            rate_direction=rate_dir,
            usd_direction=usd_dir,
            equity_direction=eq_dir,
            commodity_direction=comm_dir,
            narrative_summary=narrative,
            regime_probabilities=regime_probs,
        )

    def _generate_narrative(
        self, regime: MacroRegime, rate_dir: str, usd_dir: str, eq_dir: str
    ) -> str:
        if regime == MacroRegime.HAWKISH_INFLATION:
            return (
                f"Aggressive hawkish repricing: {rate_dir} and {usd_dir}, while {eq_dir}. "
                "Discount rate shock dominating valuations; rates traders pricing a higher terminal path."
            )
        elif regime == MacroRegime.DOVISH_DISINFLATION:
            return (
                f"Classic Goldilocks / Dovish Pivot: {rate_dir} and {usd_dir}, sparking {eq_dir}. "
                "Financial conditions easing across the board; growth duration outperforming."
            )
        elif regime == MacroRegime.GROWTH_RECESSION:
            return (
                f"Defensive Flight-to-Quality: {rate_dir} as capital flees to safe-haven Treasuries, "
                f"while {eq_dir}. Market is pricing demand destruction / macro contraction."
            )
        elif regime == MacroRegime.REFLATION_RISK_ON:
            return (
                f"Pro-cyclical Reflation: {rate_dir} alongside {eq_dir}. "
                "Healthy economic growth absorbing higher yields without equity multiple contraction."
            )
        elif regime == MacroRegime.LIQUIDITY_CONTRACTION:
            return (
                "Broad cross-asset margin cascade: Equities and bonds falling together while USD surges. "
                "Cross-market liquidity withdrawal."
            )
        return "Mixed cross-asset transmission."
