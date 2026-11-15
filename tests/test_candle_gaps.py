import pandas as pd

from inputs.collector_binance import find_candle_gaps


def test_no_missing_hourly_candles():
    timestamps = pd.to_datetime([
        "2026-01-01 10:00:00",
        "2026-01-01 11:00:00",
        "2026-01-01 12:00:00",
    ], utc=True)

    gaps = find_candle_gaps(timestamps, "1h")

    assert gaps == []


def test_detects_one_missing_hourly_candle():
    timestamps = pd.to_datetime([
        "2026-01-01 10:00:00",
        "2026-01-01 11:00:00",
        "2026-01-01 13:00:00",
    ], utc=True)

    gaps = find_candle_gaps(timestamps, "1h")

    assert len(gaps) == 1
    assert gaps[0][2] == pd.Timedelta(hours=2)


def test_detects_multiple_missing_hourly_candles():
    timestamps = pd.to_datetime([
        "2026-01-01 10:00:00",
        "2026-01-01 12:00:00",
        "2026-01-01 15:00:00",
    ], utc=True)

    gaps = find_candle_gaps(timestamps, "1h")

    assert len(gaps) == 2
    assert gaps[0][2] == pd.Timedelta(hours=2)
    assert gaps[1][2] == pd.Timedelta(hours=3)