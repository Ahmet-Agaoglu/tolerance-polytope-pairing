"""Plot cumulative standard errors for the final no-threshold experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


MODELS = ("Wide centered", "Narrow centered", "Narrow shifted")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", type=Path)
    parser.add_argument("output_png", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    batches = json.loads(args.input_json.read_text(encoding="utf-8"))

    series = {model: [] for model in MODELS}
    for batch in batches:
        by_model = {
            result["model"]: result
            for result in batch["results"]
            if result["radius_um"] == 2.0
        }
        for model in MODELS:
            series[model].append(float(by_model[model]["G"]))

    start = 5
    counts = np.arange(start, len(batches) + 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for model in MODELS:
        values = np.asarray(series[model], dtype=float)
        standard_errors = np.asarray(
            [values[:count].std(ddof=1) / np.sqrt(count) for count in counts]
        )
        ax.plot(counts, standard_errors, linewidth=1.8, label=model)

    ax.axhline(0.125, color="black", linestyle="--", linewidth=1.2,
               label="Precision target")
    ax.set_xlabel("Number of batches")
    ax.set_ylabel(r"Standard error of mean $G$")
    ax.set_xlim(start, len(batches))
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
