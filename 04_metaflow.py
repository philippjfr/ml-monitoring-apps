"""
Demo D fallback: the 03_compare.py UI, with runs read from ChurnFlow history
instead of trained at startup.

Local history (pixi run seed-runs first):
    pixi run serve-metaflow
Outerbounds production runs, the same runs whose Cards Demo G opens:
    pixi run serve-metaflow-remote
"""

import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
from metaflow import Flow, get_metadata, namespace
from sklearn.metrics import precision_score, recall_score

pn.extension(throttled=True)

namespace(None)
# The service also holds other projects' and dev branches' runs; the local store does not.
if get_metadata().startswith("service"):
    history = Flow("ChurnFlow").runs("project:churn_workshop", "project_branch:prod")
else:
    history = Flow("ChurnFlow")
runs = [run for run in history if run.successful][:10]
if not runs:
    raise RuntimeError(
        "No successful ChurnFlow runs found. Run `pixi run seed-runs` for local history, "
        "or deploy and trigger the flow for Outerbounds history."
    )

RUNS = {
    # Local run IDs are 16-digit timestamps; the tail keeps labels short but unique.
    f"…{run.id[-5:]} · {run.data.n_estimators} trees · {run.created_at:%b %d %H:%M}": {
        "metrics": run.data.metrics,
        "roc_data": run.data.roc_data,
        "predictions": run.data.predictions,
    }
    for run in runs
}
labels = list(RUNS)
# Reruns with the same n_estimators produce identical curves, so default to distinct ones.
distinct = {label.split(" · ")[1]: label for label in reversed(labels)}
default_runs = [label for label in labels if label in distinct.values()][:3]

picker = pn.ui.MultiSelect(
    label="Runs to compare",
    options=labels,
    value=default_runs,
    size=min(len(labels), 6),
)
baseline = pn.ui.Select(label="Baseline run", options=labels, value=default_runs[-1])
capacity = pn.ui.IntSlider(
    label="Weekly contact capacity", start=25, end=300, step=25, value=100
)
precision_floor = pn.ui.FloatSlider(
    label="Precision floor", start=0.0, end=1.0, step=0.01, value=0.6
)


def operating_metrics(run, capacity_value):
    # Compare at the worklist size the team can contact, not at a shared threshold:
    # a better-ranked model flags fewer customers at 0.5 and would look worse on recall.
    predictions = run["predictions"]
    worklist = predictions["churn_probability"].rank(method="first", ascending=False)
    y_pred = (worklist <= capacity_value).astype(int)
    return {
        "ROC AUC": run["metrics"]["roc_auc"],
        "Precision at capacity": precision_score(
            predictions["y_true"], y_pred, zero_division=0
        ),
        "Recall at capacity": recall_score(
            predictions["y_true"], y_pred, zero_division=0
        ),
        "Score threshold": predictions["churn_probability"][y_pred == 1].min(),
    }


def comparison_table(selected, capacity_value):
    if not selected:
        return pd.DataFrame()
    table = pd.DataFrame(
        {label: operating_metrics(RUNS[label], capacity_value) for label in selected}
    )
    return table.round(3)


def decision(selected, baseline_label, capacity_value, floor):
    base = operating_metrics(RUNS[baseline_label], capacity_value)
    lines = [f"**Baseline:** {baseline_label}"]
    for label in selected:
        if label == baseline_label:
            continue
        candidate = operating_metrics(RUNS[label], capacity_value)
        blockers = []
        if candidate["ROC AUC"] < base["ROC AUC"]:
            blockers.append("ROC AUC declined")
        if candidate["Recall at capacity"] <= base["Recall at capacity"]:
            blockers.append("recall did not improve")
        if candidate["Precision at capacity"] < floor:
            blockers.append("precision below floor")
        verdict = "promote" if not blockers else "hold: " + ", ".join(blockers)
        lines.append(f"- **{label}**: {verdict}")
    return "\n".join(lines)


def roc_overlay(selected):
    if not selected:
        return hv.NdOverlay({})
    curves = pd.concat(RUNS[label]["roc_data"].assign(run=label) for label in selected)
    return curves.hvplot.line(
        x="fpr",
        y="tpr",
        by="run",
        title="ROC: selected runs",
        xlabel="False positive rate",
        ylabel="True positive rate",
        height=360,
        responsive=True,
    )


pn.ui.Page(
    title="Churn model monitoring · ChurnFlow runs",
    sidebar=[picker, baseline, capacity, precision_floor],
    main=[
        pn.ui.Container(
            pn.ui.Markdown(
                pn.bind(decision, picker, baseline, capacity, precision_floor)
            ),
            pn.ui.DataFrame(pn.bind(comparison_table, picker, capacity)),
            hv.DynamicMap(pn.bind(roc_overlay, picker)),
        )
    ],
).servable()
