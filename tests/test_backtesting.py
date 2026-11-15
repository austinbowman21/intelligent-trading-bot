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