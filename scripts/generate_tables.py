#!/usr/bin/env python3
"""Summarize archived metrics; does not recompute model performance."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("mAP_QPIC", "mAP_rare_QPIC", "mAP_non_rare_QPIC")


def main():
    with (ROOT / "results/experiments.csv").open(newline="") as handle:
        experiments = list(csv.DictReader(handle))
    output = ROOT / "results/tables"
    output.mkdir(parents=True, exist_ok=True)
    for table in ("main", "ablation"):
        rows = []
        for experiment in experiments:
            if table not in experiment["paper_table"].split(";"):
                continue
            with (ROOT / experiment["metrics_file"]).open(newline="") as handle:
                scores = list(csv.DictReader(handle))
            if len(scores) != 1:
                raise ValueError(f"Expected one metric row: {experiment['metrics_file']}")
            rows.append({
                "experiment_id": experiment["experiment_id"],
                "configuration": experiment["label"],
                "mode": experiment["mode"],
                **{key: f"{float(scores[0][key]):.4f}" for key in METRICS},
            })
        columns = ["experiment_id", "configuration", "mode", *METRICS]
        with (output / f"{table}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        lines.extend("| " + " | ".join(row[key] for key in columns) + " |" for row in rows)
        (output / f"{table}.md").write_text("\n".join(lines) + "\n")
    print("Generated main and ablation CSV/Markdown tables from archived metrics.")


if __name__ == "__main__":
    main()
