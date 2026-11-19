# QuantNexus-Macro: Institutional Cross-Asset Macroeconomic Transmission & Dislocation Alpha Platform

[![CI / Test Suite](https://img.shields.io/badge/tests-11%20passed%20(100%25)-brightgreen.svg)]()
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)]()
[![License](https://img.shields.io/badge/license-MIT-green.svg)]()
[![Target Desks](https://img.shields.io/badge/target-Prop%20Desks%20%7C%20Macro%20Funds%20%7C%20Rates%2FFX-orange.svg)]()

---

## 1. Executive Summary & Problem Formulation

In modern electronic markets, macroeconomic data releases (**CPI, Non-Farm Payrolls, FOMC rate decisions, PPI, ISM PMI**) serve as the single largest sources of systemic volatility. When a release hits the news wires:
- Primary rates futures (**SOFR, 2Y/5Y/10Y Treasury Notes, Bunds**) and benchmark FX crosses adjust within **milliseconds to seconds**.
- Secondary assets—including equity index futures (**ES, NQ, RTY/IWM**), corporate credit (**HYG, LQD**), cross-currency basis, commodity complexes (**Gold, Crude Oil**), and yield curve spreads (**2s10s, 5s30s**)—frequently exhibit **delayed transmission, structural lag, and liquidity-vacuum overshooting**.

Standard open-source trackers (such as naive single-script tools) suffer from fatal structural flaws:
1. **Disconnected from Consensus Expectations**: They plot arbitrary intraday windows without parsing consensus estimates, whisper numbers, or standardized forecast error variance ($Z$-scores).
2. **Incorrect Event Study Econometrics**: They normalize to the beginning of a data window rather than exact $T_0$, fail to compute Cumulative Abnormal Returns (CAR), and disregard statistical significance ($t$-statistics, $p$-values).
3. **No Curve Deconstruction**: Real prop trading desks (like Future First) do not trade single assets in isolation; they trade **yield curve twists (Bear Flattener, Bull Steepener)** and butterfly curvature.
4. **Zero Actionable Alpha**: Existing tools are purely passive retrospective viewers. They have zero concept of **real-time mispricing, dislocation detection, or tradeable arbitrage signals**.

**QuantNexus-Macro** solves these fundamental gaps, transforming passive event observation into an **active quantitative statistical arbitrage and cross-market transmission engine**.

---

## 2. The Game Changer: Macro Transmission Dislocation Engine (MTDE)

### Why MTDE is a Game Changer for Prop Trading Firms (Future First, Hertshten, Jump, Tower)
Proprietary trading desks trading bond futures (ZT, ZF, ZN, ZB), equity futures (ES, NQ), and FX need to answer three questions within the first 60 seconds of a release:
1. *Given the actual economic surprise ($Z$-score), what is the theoretical fair-value price for every asset in the cross-market manifold?*
2. *Which specific assets have failed to reprice (lagging transmission), and which assets have overshot fair value due to stop-loss cascades and liquidity vacuums?*
3. *What is the statistical arbitrage trade (Entry, Convergence Target, Stop-Loss, Half-Life, and VaR risk limits)?*

### Mathematical Formulation

#### A. Standardized Macro Surprise ($Z$-Score)
$$Z_k = \frac{\text{Actual}_k - \text{Consensus}_k}{\hat{\sigma}_{\text{forecast error}, k}}$$
where $\hat{\sigma}_k$ represents the historical standard deviation of Bloomberg consensus survey forecast errors for indicator $k$.

#### B. Theoretical Implied Fair-Value Path
For any asset $i$ at minute $t$ post-release ($t \ge 0$):
$$\hat{R}_i(t) = \beta_{i,k} \cdot Z_k \cdot \Psi(t)$$
where:
- $\beta_{i,k}$ is the historical elasticity of asset $i$ to a $+1\sigma$ shock in macro indicator $k$.
- $\Psi(t) = 1 - e^{-t / \tau_{\text{abs}}}$ is the empirical information absorption function ($\tau_{\text{abs}} \approx 4.0$ minutes).
- $\hat{P}_i(t) = P_i(T_0) \cdot \left(1 + \hat{R}_i(t)\right)$.

#### C. The Macro Dislocation Index (MDI)
The residual deviation between the actual market trajectory and the theoretical macro fair-value path is standardized:
$$\epsilon_i(t) = \frac{P_i(t) - P_i(T_0)}{P_i(T_0)} - \hat{R}_i(t)$$
$$\text{MDI}_i(t) = \frac{\epsilon_i(t)}{\hat{\sigma}_{\text{intraday}, i} \cdot \sqrt{\max(t, 1) / 60}}$$

#### D. Algorithmic Trade Generation
| MDI Condition | Market State | Prop Desk Action | Execution Bracket |
|---|---|---|---|
| $\text{MDI} \le -2.50\sigma$ | **Grossly Oversold / Liquidity Cascade** | **BUY / FADE OVERSHOOT** | Target: $\hat{P}_i(t)$, Stop: $-80\text{ bps}$ |
| $-2.50\sigma < \text{MDI} \le -1.65\sigma$ | **Lagging Upside / Delayed Reaction** | **BUY CONVERGENCE** | Target: $\hat{P}_i(t)$, Stop: $-60\text{ bps}$ |
| $-1.65\sigma < \text{MDI} < +1.65\sigma$ | **Efficiently Priced** | **NEUTRAL (NO TRADE)** | Fair Value |
| $+1.65\sigma \le \text{MDI} < +2.50\sigma$ | **Lagging Downside / Unpriced Hawkishness** | **SELL CONVERGENCE** | Target: $\hat{P}_i(t)$, Stop: $+60\text{ bps}$ |
| $\text{MDI} \ge +2.50\sigma$ | **Grossly Overextended / Stop Exhaustion** | **SELL / FADE OVERSHOOT** | Target: $\hat{P}_i(t)$, Stop: $+80\text{ bps}$ |

#### E. Mean-Reversion Half-Life via Ornstein-Uhlenbeck
$$\Delta \epsilon_t = \theta (\mu - \epsilon_{t-1}) \Delta t + \sigma_{\epsilon} \Delta W_t$$
$$t_{1/2} = \frac{-\ln(2)}{\theta}$$
Provides traders with the exact expected convergence horizon in minutes.

---

## 3. Core System Architecture

```
quantnexus_macro/
│
├── config.py                  # Cross-asset universe, duration proxies, macro priors, forecast sigmas
├── engine.py                  # Master QuantNexusEngine orchestrator
├── cli.py                     # Institutional Rich terminal CLI
├── server.py                  # FastAPI real-time REST API & dashboard server
│
├── core/
│   ├── calendar.py            # Economic calendar, consensus deviations, Z-score surprises
│   ├── data_feed.py           # Real yfinance feed + Calibrated Merton Jump-Diffusion engine
│   └── event_study.py         # Campbell-Lo-MacKinlay event study, CAR, t-stats, p-values
│
├── analytics/
│   ├── curve.py               # Yield curve decomposition: 2s10s/5s30s slope, Bear/Bull Flattener
│   ├── lead_lag.py            # High-frequency cross-correlation lag profiles, T50/T80 absorption
│   ├── regime.py              # Macro Regime Classifier (Hawkish, Dovish, Growth, Reflation)
│   ├── dislocation.py         # [GAME CHANGER] MTDE Dislocation Engine & Arbitrage Signals
│   └── cohort_backtest.py     # Historical event cohorts & empirical quantile confidence cones
│
├── signals/
│   └── execution_signals.py   # Algorithmic order tickets, spread hedges, VaR 99% position sizing
│
└── visualization/
    ├── dashboard.py           # 4-Quadrant Institutional Dark-Themed Plotly Terminal
    └── report_generator.py    # Self-contained publication-grade HTML report generator
```

---

## 4. Multi-Tenor Yield Curve Decomposition

Prop bond desks trade curve spreads (**2s10s, 5s30s, 2s5s10s butterfly**). QuantNexus-Macro maps ETF price movements to implied basis point yield shifts:
$$\Delta Y_k \approx -\frac{\Delta P_k / P_k}{D_{\text{mod}, k}} \times 10,000\text{ bps}$$

Where:
- **2Y Tenor (SHY)**: $D_{\text{mod}} \approx 1.90$
- **5Y Tenor (IEI)**: $D_{\text{mod}} \approx 4.50$
- **10Y Tenor (IEF)**: $D_{\text{mod}} \approx 7.80$
- **20Y/30Y Tenor (TLT)**: $D_{\text{mod}} \approx 16.50$

### Curve Twist Regimes
1. **Bear Flattener** ($\Delta Y_2 > \Delta Y_{10} > 0, \Delta \text{Slope} < 0$): Aggressive front-end repricing; hawkish Fed rate hikes.
2. **Bear Steepener** ($\Delta Y_{10} > \Delta Y_2 > 0, \Delta \text{Slope} > 0$): Long-end yield surge; term premium or fiscal supply shock.
3. **Bull Steepener** ($\Delta Y_2 < \Delta Y_{10} < 0, \Delta \text{Slope} > 0$): Rapid front-end yield collapse; emergency easing / rate cut pricing.
4. **Bull Flattener** ($\Delta Y_{10} < \Delta Y_2 < 0, \Delta \text{Slope} < 0$): Long-end yields fall faster; growth downgrade / flight-to-safety.

---

## 5. Verification & Testing

The system is backed by a rigorous test suite (`pytest`) verifying mathematical properties, zero lookahead bias, and numerical stability:

```bash
python -m pytest -v
```

Output:
```
============================= test session starts =============================
tests/test_calendar.py::test_calendar_surprise_and_z_score PASSED        [  9%]
tests/test_calendar.py::test_calendar_repository PASSED                  [ 18%]
tests/test_cohort_and_e2e.py::test_confidence_cone_properties PASSED     [ 27%]
tests/test_cohort_and_e2e.py::test_end_to_end_engine_pipeline PASSED     [ 36%]
tests/test_curve.py::test_yield_curve_shifts_and_regimes PASSED          [ 45%]
tests/test_data_feed.py::test_jump_diffusion_calibration PASSED          [ 54%]
tests/test_dislocation.py::test_dislocation_engine PASSED                [ 63%]
tests/test_dislocation.py::test_execution_ticket_generation PASSED       [ 72%]
tests/test_event_study.py::test_event_study_metrics PASSED               [ 81%]
tests/test_lead_lag.py::test_lead_lag_analyzer PASSED                    [ 90%]
tests/test_lead_lag.py::test_regime_classifier PASSED                    [100%]
============================= 11 passed in 6.33s ==============================
```

---

## 6. Quickstart Guide

### Installation
```bash
git clone <repo-url>
cd QuantNexus-Macro
pip install -r requirements.txt
pip install -e .
```

### CLI Terminal Usage
```bash
# List all registered historical benchmark events
python -m quantnexus_macro.cli list-events

# Run full cross-asset transmission analysis for a specific shock
python -m quantnexus_macro.cli analyze --event CPI_2024_02_13

# Force calibrated jump-diffusion simulation mode for offline/stress-testing
python -m quantnexus_macro.cli analyze --event CPI_2024_02_13 --sim
```

### Programmatic Python API
```python
from quantnexus_macro.engine import QuantNexusEngine

engine = QuantNexusEngine(book_capital_usd=10_000_000.0)

# Run complete institutional event analysis
pkg = engine.analyze_event(
    event_or_id="CPI_2024_02_13",
    pre_event_minutes=30,
    post_event_minutes=120,
    generate_report=True,
)

print(f"Detected Regime: {pkg.regime_result.detected_regime.value}")
print(f"2s10s Curve Slope: {pkg.curve_result.slope_2s10s_bps_change:+.1f} bps")
print(f"Market Efficiency Score: {pkg.dislocation_result.cross_asset_efficiency_score}%")

# Inspect top dislocation trade
top_trade = pkg.dislocation_result.highest_alpha_opportunity
if top_trade:
    print(f"Signal: {top_trade.action.value} on {top_trade.ticker}")
    print(f"Entry: ${top_trade.current_price:.2f} -> Target: ${top_trade.target_price:.2f}")
    print(f"Dislocation Z: {top_trade.macro_dislocation_index:+.2f}σ | R:R: {top_trade.risk_reward_ratio:.2f}")

print(f"Interactive HTML Report: {pkg.report_html_path}")
```

### Interactive Web Service & REST API
```bash
uvicorn quantnexus_macro.server:app --reload --port 8000
```
- Interactive Portal: `http://localhost:8000/`
- Full Analysis JSON API: `http://localhost:8000/api/analyze/CPI_2024_02_13`
- Standalone HTML Report: `http://localhost:8000/api/report/CPI_2024_02_13`

---

## 7. Institutional Impact for Prop Firms

| Dimension | Naive Trackers (Reference Repo) | QuantNexus-Macro |
|---|---|---|
| **Consensus & Surprises** | None (Ignores actuals & forecasts) | Standardized $Z$-Score with Bloomberg forecast sigmas |
| **Normalization** | Arbitrary window start | Exact $T_0 \equiv 100.0$ synchronization |
| **Statistical Rigor** | None (Mock random walks) | Cumulative Abnormal Returns (CAR), $t$-stats, $p$-values |
| **Fixed Income / Rates** | Single long bond (TLT) | Multi-Tenor Curve (2Y, 5Y, 10Y, 20Y+), 2s10s/5s30s slope, Bear/Bull Flattener |
| **Microstructural Lead-Lag** | None | Cross-correlation lag profiles, T50/T80 absorption speeds |
| **Regime Identification** | None | 5-Regime Concordance Matrix (Hawkish, Dovish, Growth, Reflation, Liquidity) |
| **Actionable Alpha (MTDE)** | **None (Purely passive)** | **Real-time Dislocation Index (MDI), Convergence & Overshoot Arbitrage** |
| **Execution Sizing** | None | Algorithmic blotters, duration-matched hedges, 99% 1-hour VaR bounds |
| **Historical Context** | None | Empirical quantile cones (10th, 25th, 50th, 75th, 90th percentile fan charts) |

---

## 8. License & Attribution
Developed for institutional quantitative research and proprietary trading systems. Released under the MIT License.
