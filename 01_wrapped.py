"""
Demo A fallback: saved evaluation results wrapped in a Panel layout, unchanged.

    pixi run serve-wrapped
"""

import json
from pathlib import Path

import pandas as pd
import panel as pn

pn.extension()

ARTIFACTS = Path(__file__).parent / "artifacts"

metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
metrics_df = pd.DataFrame([metrics])
predictions_df = pd.read_parquet(ARTIFACTS / "predictions.parquet")
confusion_matrix_pane = pn.pane.Image(ARTIFACTS / "confusion_matrix.png", width=350)

pn.ui.Column(
    "# Churn model · saved evaluation results",
    "## Metrics",
    pn.ui.DataFrame(metrics_df, index=False),
    "## Confusion matrix",
    confusion_matrix_pane,
    "## Predictions (first 20)",
    pn.ui.DataFrame(predictions_df.head(20), index=False),
).servable()
