import pandas as pd

from common.backtesting import simulated_trade_performance


def test_long_trade_profit():
    df = pd.DataFrame({
        "buy": [True, False],
        "sell": [False, True],
        "close": [100, 110],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(df, "buy", "sell", "close")
    )

    assert long_performance["#transactions"] == 1
    assert long_performance["profit"] == 10.00
    assert long_performance["%profit"] == 10.0


def test_short_trade_profit():
    df = pd.DataFrame({
        "buy": [False, True],
        "sell": [True, False],
        "close": [100, 90],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(df, "buy", "sell", "close")
    )

    assert short_performance["#transactions"] == 1
    assert short_performance["profit"] == 10.00
    assert short_performance["%profit"] == 10.0
def test_long_direction_ignores_sell_signal_when_no_position():
    df = pd.DataFrame({
        "buy": [False, True, False],
        "sell": [True, False, True],
        "close": [100, 105, 110],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(
            df, "buy", "sell", "close", direction="long"
        )
    )

    assert long_performance["#transactions"] == 1
    assert long_performance["profit"] == 5.00
    assert short_performance["#transactions"] == 0


def test_short_direction_ignores_buy_signal_when_no_position():
    df = pd.DataFrame({
        "buy": [True, False, True],
        "sell": [False, True, False],
        "close": [100, 105, 100],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(
            df, "buy", "sell", "close", direction="short"
        )
    )

    assert short_performance["#transactions"] == 1
    assert short_performance["profit"] == 5.00
    assert long_performance["#transactions"] == 0


def test_open_long_position_is_reported_separately():
    df = pd.DataFrame({
        "buy": [True, False],
        "sell": [False, False],
        "close": [100, 110],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(df, "buy", "sell", "close")
    )

    assert performance["#transactions"] == 0
    assert performance["profit"] == 0.0
    assert long_performance["open_position"]["side"] == "long"
    assert long_performance["open_position"]["unrealized_profit"] == 10.0
    assert short_performance["open_position"] is None


def test_open_short_position_is_reported_separately():
    df = pd.DataFrame({
        "buy": [False, False],
        "sell": [True, False],
        "close": [100, 90],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(df, "buy", "sell", "close")
    )

    assert performance["#transactions"] == 0
    assert performance["profit"] == 0.0
    assert short_performance["open_position"]["side"] == "short"
    assert short_performance["open_position"]["unrealized_profit"] == 10.0
    assert long_performance["open_position"] is None


def test_open_long_position_estimate_includes_fees_and_slippage():
    df = pd.DataFrame({
        "buy": [True, False],
        "sell": [False, False],
        "close": [100, 110],
    })

    performance, long_performance, short_performance = (
        simulated_trade_performance(
            df, "buy", "sell", "close", fee_rate=0.001, slippage_rate=0.001
        )
    )

    assert long_performance["open_position"]["unrealized_profit"] == 9.58
    assert long_performance["open_position"]["estimated_exit_price"] == 109.89
    assert performance["#transactions"] == 0
