#!/usr/bin/env python3
"""Aggregate OpenCompass summaries using the paper's macro averaging."""

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean


MMLU_PREFIX = "lukaemon_mmlu_"


def metric(rows, dataset, name, model):
    for row in rows:
        if row["dataset"] == dataset and row["metric"] == name:
            return float(row[model])
    raise KeyError(f"Missing {dataset}/{name} for {model}")


def summarize(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        models = [
            column
            for column in (reader.fieldnames or [])
            if column not in {"dataset", "version", "metric", "mode"}
        ]

    mmlu_datasets = sorted(
        {
            row["dataset"]
            for row in rows
            if row["dataset"].startswith(MMLU_PREFIX)
        }
    )
    if len(mmlu_datasets) != 19:
        raise ValueError(
            f"{path}: expected 19 MMLU STEM domains, found {len(mmlu_datasets)}"
        )

    output = []
    for model in models:
        domain_accuracy = [
            100.0
            * metric(rows, dataset, "correct_count", model)
            / metric(rows, dataset, "total_count", model)
            for dataset in mmlu_datasets
        ]
        domain_var = []
        for dataset in mmlu_datasets:
            total = metric(rows, dataset, "total_count", model)
            invalid = metric(rows, dataset, "no_answer_count", model) + metric(
                rows, dataset, "multiple_answers_count", model
            )
            domain_var.append(100.0 * (total - invalid) / total)

        correct = sum(
            metric(rows, dataset, "correct_count", model)
            for dataset in mmlu_datasets
        )
        total = sum(
            metric(rows, dataset, "total_count", model)
            for dataset in mmlu_datasets
        )
        invalid = sum(
            metric(rows, dataset, "no_answer_count", model)
            + metric(rows, dataset, "multiple_answers_count", model)
            for dataset in mmlu_datasets
        )
        output.append(
            {
                "source": str(path),
                "model": model,
                "paper_metrics": {
                    "arc_easy_accuracy": metric(
                        rows, "ARC-e", "score", model
                    ),
                    "arc_challenge_accuracy": metric(
                        rows, "ARC-c", "score", model
                    ),
                    "gpqa_diamond_accuracy": metric(
                        rows, "GPQA_diamond", "score", model
                    ),
                    "mmlu_stem_accuracy_macro": fmean(domain_accuracy),
                    "mmlu_stem_var_macro": fmean(domain_var),
                },
                "diagnostic_micro_metrics": {
                    "mmlu_stem_accuracy_micro": 100.0 * correct / total,
                    "mmlu_stem_var_micro": 100.0 * (total - invalid) / total,
                },
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        type=Path,
        help="Summary CSV or directory containing summary_*.csv files.",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    files = (
        [args.path]
        if args.path.is_file()
        else sorted(args.path.rglob("summary_*.csv"))
    )
    if not files:
        parser.error(f"no summary_*.csv files found under {args.path}")
    rendered = json.dumps(
        [item for path in files for item in summarize(path)], indent=2
    ) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
