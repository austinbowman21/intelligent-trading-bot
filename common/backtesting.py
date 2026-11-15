
import numpy as np
import pandas as pd

"""
Backtesting and trade performance using trade simulation.
"""


def simulated_trade_performance(
    df,
    buy_signal_column,
    sell_signal_column,
    price_column,
    direction="both",
    fee_rate=0.0,
    slippage_rate=0.0,
):
    """
    Simulate long, short, or both types of trades.

    Fees and slippage are decimal rates applied to each execution.
    Only completed trades are included in performance statistics.
    """

    if direction not in ("long", "short", "both"):
        raise ValueError("direction must be 'long', 'short', or 'both'.")

    for name, rate in (
        ("fee_rate", fee_rate),
        ("slippage_rate", slippage_rate),
    ):
        if not np.isfinite(rate) or rate < 0 or rate >= 1:
            raise ValueError(f"{name} must be between 0 and 1.")

    position = None
    entry_price = None
    last_valid_price = None

    long_profit = 0.0
    long_profit_percent = 0.0
    long_transactions = 0
    long_profitable = 0

    short_profit = 0.0
    short_profit_percent = 0.0
    short_transactions = 0
    short_profitable = 0

    columns = [buy_signal_column, sell_signal_column, price_column]

    for buy_signal, sell_signal, price in df[columns].itertuples(
        index=False, name=None
    ):
        if pd.isna(price):
            continue

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if not np.isfinite(price) or price <= 0:
            continue

        last_valid_price = price

        buy_signal = bool(buy_signal) if pd.notna(buy_signal) else False
        sell_signal = bool(sell_signal) if pd.notna(sell_signal) else False

        if buy_signal and sell_signal:
            continue

        if position is None:
            if buy_signal and direction in ("long", "both"):
                position = "long"
                entry_price = price * (1 + slippage_rate)

            elif sell_signal and direction in ("short", "both"):
                position = "short"
                entry_price = price * (1 - slippage_rate)

        elif position == "long" and sell_signal:
            exit_price = price * (1 - slippage_rate)

            gross_profit = exit_price - entry_price
            fees = fee_rate * (entry_price + exit_price)
            profit = gross_profit - fees
            profit_percent = 100.0 * profit / entry_price

            long_profit += profit
            long_profit_percent += profit_percent
            long_transactions += 1

            if profit > 0:
                long_profitable += 1

            position = None
            entry_price = None

        elif position == "short" and buy_signal:
            exit_price = price * (1 + slippage_rate)

            gross_profit = entry_price - exit_price
            fees = fee_rate * (entry_price + exit_price)
            profit = gross_profit - fees
            profit_percent = 100.0 * profit / entry_price

            short_profit += profit
            short_profit_percent += profit_percent
            short_transactions += 1

            if profit > 0:
                short_profitable += 1

            position = None
            entry_price = None

    def make_performance(profit, profit_percent, transactions, profitable):
        return {
            "#transactions": transactions,
            "profit": round(profit, 2),
            "%profit": round(profit_percent, 1),
            "#profitable": profitable,
            "%profitable": round(
                100.0 * profitable / transactions, 1
            ) if transactions else 0.0,
            "profit/T": round(
                profit / transactions, 2
            ) if transactions else 0.0,
            "%profit/T": round(
                profit_percent / transactions, 1
            ) if transactions else 0.0,
        }

    long_performance = make_performance(
        long_profit, long_profit_percent, long_transactions, long_profitable
    )

    short_performance = make_performance(
        short_profit, short_profit_percent, short_transactions, short_profitable
    )

    # Report any position still open at the end without counting it as a
    # completed trade. The estimate assumes it is closed at the last valid
    # observed price, including adverse exit slippage and the exit fee.
    open_position = None
    if position is not None and last_valid_price is not None:
        if position == "long":
            estimated_exit_price = last_valid_price * (1 - slippage_rate)
            gross_unrealized_profit = estimated_exit_price - entry_price
        else:
            estimated_exit_price = last_valid_price * (1 + slippage_rate)
            gross_unrealized_profit = entry_price - estimated_exit_price

        estimated_fees = fee_rate * (entry_price + estimated_exit_price)
        unrealized_profit = gross_unrealized_profit - estimated_fees
        open_position = {
            "side": position,
            "entry_price": round(entry_price, 8),
            "mark_price": round(last_valid_price, 8),
            "estimated_exit_price": round(estimated_exit_price, 8),
            "unrealized_profit": round(unrealized_profit, 2),
            "unrealized_profit_percent": round(
                100.0 * unrealized_profit / entry_price, 2
            ),
        }

    long_performance["open_position"] = (
        open_position if position == "long" else None
    )
    short_performance["open_position"] = (
        open_position if position == "short" else None
    )

    total_transactions = long_transactions + short_transactions
    total_profitable = long_profitable + short_profitable

    performance = make_performance(
        long_profit + short_profit,
        long_profit_percent + short_profit_percent,
        total_transactions,
        total_profitable,
    )

    performance["profitable"] = (
        total_profitable / total_transactions
        if total_transactions else 0.0
    )
    performance["profitable_percent"] = round(
        100.0 * performance["profitable"], 1
    )
    performance["open_position"] = open_position

    return performance, long_performance, short_performance
