import csv

import pytest

from scripts.simulate import write_simulation_results


def test_simulation_results_csv_includes_open_position_metrics(tmp_path):
    output_path = tmp_path / "signal_models.txt"
    performances = [{
        "model": {"buy_signal_threshold": 0.05},
        "performance": {
            "#transactions": 2,
            "profit": 12.5,
            "open_position_side": "long",
            "open_position_entry_price": 100.0,
            "open_position_unrealized_profit": 3.25,
        },
    }]

    assert write_simulation_results(performances, output_path) == 1

    with output_path.open("r", newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))

    assert rows[0]["open_position_side"] == "long"
    assert rows[0]["open_position_entry_price"] == "100.0"
    assert rows[0]["open_position_unrealized_profit"] == "3.25"


def test_simulation_results_csv_appends_without_duplicate_header(tmp_path):
    output_path = tmp_path / "signal_models.txt"
    performances = [{
        "model": {"threshold": 0.05},
        "performance": {"profit": 2.0, "open_position_side": ""},
    }]

    write_simulation_results(performances, output_path)
    write_simulation_results(performances, output_path)

    with output_path.open("r", newline="", encoding="utf-8") as output_file:
        rows = list(csv.DictReader(output_file))

    assert len(rows) == 2
    assert rows[0]["profit"] == "2.0"
    assert rows[1]["profit"] == "2.0"


def test_simulation_results_csv_rejects_incompatible_existing_header(tmp_path):
    output_path = tmp_path / "signal_models.txt"
    output_path.write_text("old_column\nold_value\n", encoding="utf-8")
    performances = [{
        "model": {"new_column": 0.05},
        "performance": {"profit": 2.0},
    }]

    with pytest.raises(ValueError, match="different columns"):
        write_simulation_results(performances, output_path)
