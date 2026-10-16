"""
Demo C fallback: compare a baseline against candidate runs at the weekly contact capacity.

Trains three variants at startup on ChurnFlow's fixed cohort and split, so the only
difference between runs is n_estimators. 04_metaflow.py keeps this UI and replaces
only the run loading with ChurnFlow history.

    pixi run serve-compare
"""

import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from flows.main.flow import make_churn_dataset

pn.extension(throttled=True)

NUMERIC_COLS = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets",
    "avg_session_minutes",
]
CATEGORICAL_COLS = ["contract_type"]


def train_run(n_estimators, seed=42):
    df = make_churn_dataset(random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        df[NUMERIC_COLS + CATEGORICAL_COLS],
        df["churned"],
        test_size=0.25,
        random_state=42,
    )
    pipeline = Pipeline(
        steps=[
            (
                "preprocess",
                ColumnTransformer(
                    transformers=[
                        ("num", "passthrough", NUMERIC_COLS),
                        (
                            "cat",
                            OneHotEncoder(handle_unknown="ignore"),
                            CATEGORICAL_COLS,
                        ),
                    ]
                ),
            ),
            (
                "model",
                RandomForestClassifier(n_estimators=n_estimators, random_state=seed),
            ),
        ]
    )
    pipeline.fit(X_train, y_train)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    return {
        "metrics": {"roc_auc": roc_auc_score(y_test, y_proba)},
        "roc_data": pd.DataFrame({"fpr": fpr, "tpr": tpr}),
        "predictions": pd.DataFrame(
            {"y_true": y_test.to_numpy(), "churn_probability": y_proba}
        ),
    }


RUNS = {f"{n} trees": train_run(n) for n in (10, 100, 300)}
labels = list(RUNS)

picker = pn.ui.MultiSelect(
    label="Runs to compare", options=labels, value=labels, size=len(labels)
)
baseline = pn.ui.Select(label="Baseline run", options=labels, value=labels[1])
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
    title="Churn model monitoring · compare runs",
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
