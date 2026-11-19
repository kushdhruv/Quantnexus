"""
QuantNexus-Macro: Command Line Interface (CLI)
Institutional Terminal Tool for Prop Desks and Quantitative Traders.
"""

import argparse
import sys
from datetime import datetime
import pandas as pd

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.text import Text
    console = Console()
except ImportError:
    console = None

from quantnexus_macro.engine import QuantNexusEngine
from quantnexus_macro.config import MacroEventType
from quantnexus_macro.core.calendar import MacroRelease


def print_banner():
    title = """
  ██████╗ ██╗   ██╗ █████╗ ███╗   ██╗████████╗███╗   ██╗███████╗██╗  ██╗██╗   ██╗███████╗
 ██╔═══██╗██║   ██║██╔══██╗████╗  ██║╚══██╔══╝████╗  ██║██╔════╝╚██╗██╔╝██║   ██║██╔════╝
 ██║   ██║██║   ██║███████║██╔██╗ ██║   ██║   ██╔██╗ ██║█████╗   ╚███╔╝ ██║   ██║███████╗
 ██║▄▄ ██║██║   ██║██╔══██║██║╚██╗██║   ██║   ██║╚██╗██║██╔══╝   ██╔██╗ ██║   ██║╚════██║
 ╚██████╔╝╚██████╔╝██║  ██║██║ ╚████║   ██║   ██║ ╚████║███████╗██╔╝ ██╗╚██████╔╝███████║
  ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═══╝   ╚═╝   ╚═╝  ╚═══╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝
    Institutional Macro Transmission & Cross-Asset Dislocation Engine (v1.0.0)
    Designed for Prop Desks, Macro Hedge Funds, and Rates/FX Market Makers
    """
    if console:
        console.print(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(title)


def handle_list_events(engine: QuantNexusEngine):
    """List all events currently in the calendar."""
    df = engine.calendar.to_dataframe()
    if console:
        table = Table(title="[bold yellow]Registered Macroeconomic Releases[/bold yellow]")
        table.add_column("Event ID", style="cyan", no_wrap=True)
        table.add_column("Indicator Name", style="white")
        table.add_column("Release Timestamp (UTC)", style="magenta")
        table.add_column("Actual", justify="right", style="green")
        table.add_column("Consensus", justify="right", style="blue")
        table.add_column("Surprise Z", justify="right", style="bold red")
        table.add_column("Policy Bias", style="yellow")

        for _, row in df.iterrows():
            table.add_row(
                row["event_id"],
                row["name"],
                str(row["timestamp_utc"]),
                f"{row['actual']}{row['unit']}",
                f"{row['consensus']}{row['unit']}",
                f"{row['z_score']:+.2f}σ",
                row["policy_bias"],
            )
        console.print(table)
    else:
        print(df.to_string())


def handle_analyze(args, engine: QuantNexusEngine):
    """Run full quantitative macro transmission analysis."""
    print_banner()

    if console:
        console.print(f"[bold green]Initiating Analysis for Event: {args.event}[/bold green]")
    else:
        print(f"Initiating Analysis for Event: {args.event}")

    try:
        pkg = engine.analyze_event(
            event_or_id=args.event,
            pre_event_minutes=args.pre,
            post_event_minutes=args.post,
            force_simulation=args.sim,
            generate_report=True,
        )
    except Exception as e:
        if console:
            console.print(f"[bold red]Execution Error: {e}[/bold red]")
        else:
            print(f"Execution Error: {e}")
        return

    # 1. Macro Summary Card
    event = pkg.event
    regime = pkg.regime_result
    curve = pkg.curve_result
    disloc = pkg.dislocation_result

    if console:
        console.print(
            Panel(
                f"[bold white]{event.name}[/bold white]\n"
                f"Release Time: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S UTC')}\n"
                f"Actual: [bold]{event.actual}{event.unit}[/bold] | Consensus: [bold]{event.consensus}{event.unit}[/bold] | Surprise: [bold red]{event.surprise:+.2f}{event.unit} ({event.z_score:+.2f}σ)[/bold red]\n"
                f"Policy Bias: [yellow]{event.policy_bias}[/yellow]\n\n"
                f"[bold cyan]Detected Regime:[/bold cyan] {regime.detected_regime.value}\n"
                f"[bold cyan]Regime Confidence:[/bold cyan] {int(regime.confidence_score * 100)}%\n"
                f"[bold cyan]Narrative:[/bold cyan] {regime.narrative_summary}\n\n"
                f"[bold cyan]Yield Curve Twist (2s10s):[/bold cyan] {curve.slope_2s10s_bps_change:+.1f} bps ({curve.regime.value})\n"
                f"[bold cyan]Market Efficiency Score:[/bold cyan] {disloc.cross_asset_efficiency_score}%\n"
                f"[bold cyan]Interactive HTML Dossier Generated:[/bold cyan] [underline green]{pkg.report_html_path}[/underline green]",
                title="[bold yellow]Institutional Macro Transmission Summary[/bold yellow]",
                border_style="cyan",
            )
        )

        # 2. Dislocation & Alpha Signals Table
        sig_table = Table(title="[bold red]Macro Transmission Dislocation (MTDE) Alpha Blotter[/bold red]")
        sig_table.add_column("Ticker", style="cyan")
        sig_table.add_column("Signal", style="bold")
        sig_table.add_column("Current Px", justify="right")
        sig_table.add_column("Fair Px", justify="right")
        sig_table.add_column("Disloc %", justify="right")
        sig_table.add_column("MDI (Z)", justify="right", style="bold yellow")
        sig_table.add_column("Target Px", justify="right", style="green")
        sig_table.add_column("Stop Px", justify="right", style="red")
        sig_table.add_column("R:R", justify="right")
        sig_table.add_column("Win Prob", justify="right")

        for s in disloc.signals:
            sig_col = "green" if "BUY" in s.action.value else ("red" if "SELL" in s.action.value else "white")
            sig_table.add_row(
                s.ticker,
                f"[{sig_col}]{s.action.value.split('(')[0].strip()}[/{sig_col}]",
                f"${s.current_price:.2f}",
                f"${s.implied_fair_value_price:.2f}",
                f"{s.dislocation_pct:+.2f}%",
                f"{s.macro_dislocation_index:+.2f}σ",
                f"${s.target_price:.2f}",
                f"${s.stop_loss_price:.2f}",
                f"{s.risk_reward_ratio:.2f}",
                f"{s.confidence_pct:.0f}%",
            )
        console.print(sig_table)

        # 3. Execution Tickets
        if pkg.execution_tickets:
            tkt_table = Table(title="[bold green]Prop Desk Execution Order Tickets ($10M Book)[/bold green]")
            tkt_table.add_column("Ticket ID", style="cyan")
            tkt_table.add_column("Strategy")
            tkt_table.add_column("Order", style="bold")
            tkt_table.add_column("Hedge", style="magenta")
            tkt_table.add_column("Notional ($)", justify="right")
            tkt_table.add_column("Target PnL", justify="right", style="green")
            tkt_table.add_column("Max Risk", justify="right", style="red")

            for t in pkg.execution_tickets:
                hedge_desc = f"{t.hedged_action} {t.hedged_ticker} ({t.hedge_ratio:.1f}x)" if t.hedged_ticker else "None"
                tkt_table.add_row(
                    t.ticket_id,
                    t.strategy_type,
                    f"{t.action} {t.primary_ticker}",
                    hedge_desc,
                    f"${t.suggested_notional_usd:,.0f}",
                    f"+${t.expected_pnl_usd:,.0f}",
                    f"-${t.max_loss_usd:,.0f}",
                )
            console.print(tkt_table)

    else:
        print("\n=== SUMMARY ===")
        print(f"Event: {event.name} | Surprise: {event.surprise:+.2f} ({event.z_score:+.2f}σ)")
        print(f"Regime: {regime.detected_regime.value}")
        print(f"Curve: {curve.regime.value} ({curve.slope_2s10s_bps_change:+.1f} bps)")
        print(f"Report: {pkg.report_html_path}")


def main():
    parser = argparse.ArgumentParser(
        description="QuantNexus-Macro: Institutional Cross-Asset Transmission & Dislocation Platform"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Command: list-events
    subparsers.add_parser("list-events", help="List registered historical benchmark events")

    # Command: analyze
    analyze_parser = subparsers.add_parser("analyze", help="Run full cross-asset event study & dislocation analysis")
    analyze_parser.add_argument("--event", type=str, default="CPI_2024_02_13", help="Event ID from calendar")
    analyze_parser.add_argument("--pre", type=int, default=30, help="Pre-event window in minutes")
    analyze_parser.add_argument("--post", type=int, default=120, help="Post-event window in minutes")
    analyze_parser.add_argument("--sim", action="store_true", help="Force calibrated jump-diffusion simulation")

    args = parser.parse_args()

    engine = QuantNexusEngine()

    if args.command == "list-events":
        handle_list_events(engine)
    elif args.command == "analyze" or args.command is None:
        if args.command is None:
            # Default behavior if run with no args
            args.event = "CPI_2024_02_13"
            args.pre = 30
            args.post = 120
            args.sim = False
        handle_analyze(args, engine)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
