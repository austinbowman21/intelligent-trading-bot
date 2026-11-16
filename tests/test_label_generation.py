import pytest
import numpy.testing as npt

from common.utils import *
from common.gen_signals import *
from common.gen_labels_topbot import *

def test_extremum_labels():
    data = [10, 30, 50, 70, 90, 70, 50, 30, 9]
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data)
    sr = pd.Series(data * 2)
    level_frac = 0.5
    tolerance_frac = 0.1

    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)
    minimums = find_all_extremums(sr, False, level_frac, tolerance_frac)

    # Merge into a sequence of interleaving minimums and maximums
    all = list()
    all.extend(maximums)
    all.extend(minimums)

    all.sort(key=lambda x: x[1])

    # We have indexes (x coordinates) and need to find their y values
    extr_x = [x[1] for x in all]
    extr_y = [sr[x] for x in extr_x]
    extr_df = pd.DataFrame({'x': extr_x, 'y': extr_y})

    import seaborn as sns
    # https://matplotlib.org/stable/api/markers_api.html
    # '-', '--', '-.', ':', 'None', ' ', '', 'solid', 'dashed', 'dashdot', 'dotted'
    sns.lineplot(data=sr)
    sns.lineplot(data=extr_df, x="x", y="y", marker="o", markersize=10, linestyle='dotted')  # "^" 'v'

    pass

def test_interval_and_aggregation():
    data = [10, 40, 30, 70, 90, 50, 60, 30, 9]
    sr = pd.Series(data * 2)

    df = pd.DataFrame(data={'close': sr, 'score': sr / 10})

    level_frac = 0.5
    tolerance_frac = 0.1

    # For debugging
    maximums = find_all_extremums(sr, True, level_frac, tolerance_frac)

    # Add label
    df, _ = add_extremum_features(df, column_name='close', level_fracs=[level_frac], tolerance_frac=tolerance_frac, out_names=['is_close_top'])

    # Aggregate score with chosen parameters
    aggregate_scores(df, 'score_agg', ['score'], None, 2)

    threshold = 6
    interval_df = find_interval_precision(df, label_column='is_close_top', score_column='score_agg', threshold=threshold)

    pass


def test_crossing_threshold_requires_complete_future_window():
    from common.utils import first_location_of_crossing_threshold

    df = pd.DataFrame({
        "close": [100.0, 100.0, 100.0, 100.0, 100.0],
        "high": [101.0, np.nan, 104.0, 101.0, 101.0],
    })

    result = first_location_of_crossing_threshold(df, 4, 3, "close", "high")

    # The missing first future candle makes the first-cross location unknowable.
    assert pd.isna(result.iloc[0])
    # A complete window with no crossing is also represented as NaN.
    assert pd.isna(result.iloc[1])


def test_first_cross_label_marks_missing_future_candle_unknown():
    from common.gen_labels_highlow import first_cross_labels

    df = pd.DataFrame({
        "close": [100.0] * 5,
        "high": [101.0, np.nan, 104.0, 101.0, 101.0],
        "low": [99.0] * 5,
    })

    first_cross_labels(df, 4, [3.0, -0.6], "close", ["high", "low"], "high_30")

    assert pd.isna(df.loc[0, "high_30"])


def test_first_cross_label_marks_same_candle_tie_unknown():
    from common.gen_labels_highlow import first_cross_labels

    df = pd.DataFrame({
        "close": [100.0, 100.0, 100.0],
        "high": [100.0, 104.0, 100.0],
        "low": [100.0, 99.0, 100.0],
    })

    first_cross_labels(df, 2, [3.0, -0.6], "close", ["high", "low"], "high_30")

    assert pd.isna(df.loc[0, "high_30"])


def test_first_cross_label_marks_incomplete_tail_unknown():
    from common.gen_labels_highlow import first_cross_labels

    df = pd.DataFrame({
        "close": [100.0] * 5,
        "high": [101.0] * 5,
        "low": [99.0] * 5,
    })

    first_cross_labels(df, 2, [3.0, -0.6], "close", ["high", "low"], "high_30")

    assert pd.notna(df.loc[0, "high_30"])
    assert pd.isna(df.loc[3, "high_30"])
    assert pd.isna(df.loc[4, "high_30"])
