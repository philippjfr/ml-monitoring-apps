"""
Demo B fallback: adds a decision-threshold slider, a segment filter, and
global feature importance, laid out in a pn.ui.Page.

    pixi run serve-monitoring
"""

from pathlib import Path

import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
from sklearn.metrics import confusion_matrix, roc_curve

pn.extension(throttled=True)

ARTIFACTS = Path(__file__).parent / "artifacts"

predictions_df = pd.read_parquet(ARTIFACTS / "predictions.parquet")
feature_importance = pd.read_csv(ARTIFACTS / "feature_importance.csv")

threshold = pn.ui.FloatSlider(
    label="Decision threshold", start=0.0, end=1.0, step=0.01, value=0.5
)
segment = pn.ui.Select(
    label="Segment",
    options=["All"] + sorted(predictions_df["contract_type"].unique().tolist()),
    value="All",
)


def _filtered(segment_value):
    if segment_value == "All":
        return predictions_df
    return predictions_df[predictions_df["contract_type"] == segment_value]


def _confusion_table(threshold_value, segment_value):
    subset = _filtered(segment_value)
    y_pred = (subset["churn_probability"] >= threshold_value).astype(int)
    cm = confusion_matrix(subset["y_true"], y_pred, labels=[0, 1])
    return pd.DataFrame(
        cm,
        index=pd.Index(["actually stayed", "actually churned"], name="actual"),
        columns=pd.Index(["predicted stayed", "predicted churned"], name="prediction"),
    )


def worklist_summary(threshold_value, segment_value):
    subset = _filtered(segment_value)
    worklist_size = int((subset["churn_probability"] >= threshold_value).sum())
    return f"**Retention worklist:** {worklist_size} customers. Capacity: 100 customers per week."


def prediction_distribution(segment_value):
    return _filtered(segment_value).hvplot.hist(
        y="churn_probability",
        bins=20,
        title="Prediction distribution",
        xlabel="Churn probability",
        ylabel="Customers",
        color="#08ca4a",
        height=320,
        responsive=True,
    )


def roc_plot(segment_value):
    subset = _filtered(segment_value)
    fpr, tpr, _ = roc_curve(subset["y_true"], subset["churn_probability"])
    roc_data = pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr})
    return roc_data.hvplot.line(
        x="false_positive_rate",
        y="true_positive_rate",
        title="ROC curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        color="#08ca4a",
        height=320,
        responsive=True,
    )


def feature_importance_plot():
    return feature_importance.head(8).iloc[::-1].hvplot.barh(
        x="feature",
        y="importance",
        title="Feature importance",
        color="#08ca4a",
        height=320,
        responsive=True,
    )


sidebar = [threshold, segment]
summary_ref = pn.bind(worklist_summary, threshold, segment)
table_ref = pn.bind(_confusion_table, threshold, segment)
distribution_ref = pn.bind(prediction_distribution, segment)
roc_ref = pn.bind(roc_plot, segment)
main = [
    pn.ui.Column(
        pn.ui.Typography("Threshold evaluation", variant="h5"),
        pn.ui.Markdown(summary_ref),
        pn.ui.DataFrame(table_ref),
        pn.ui.Row(hv.DynamicMap(distribution_ref), hv.DynamicMap(roc_ref)),
    ),
    feature_importance_plot(),
]

pn.ui.Page(
    title="Churn model monitoring",
    sidebar=sidebar,
    main=main,
).servable()
