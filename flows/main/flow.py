"""
ChurnFlow — Metaflow pipeline for the Session 4 workshop.

Trains the synthetic churn model and stores evaluation artifacts for the
Panel app. The notebook imports make_churn_dataset for EDA without training.
Production runs register the model asset and render Cards in evaluate.

Run:
    pixi run flow --n-estimators 200 --seed 42
"""

import numpy as np
import pandas as pd
from metaflow import Parameter, card, current, step
from obproject import ProjectFlow

try:
    from _env import step_env
except Exception:
    step_env = lambda *args, **kwargs: (lambda fn: fn)  

DEPS = {
    "numpy": "2.4.4",
    "pandas": "2.3.3",
    "scikit-learn": "1.8.0",
    "pyarrow": "24.0.0",
}
EVAL_DEPS = {**DEPS, "shap": "0.50.0", "metaflow-card-html": "1.0.3"}

IFRAME_TEMPLATE = """<!doctype html>
<html>
  <head><meta charset="utf-8"><style>html,body,iframe{{margin:0;border:0;width:100%;height:100%;overflow:hidden}}</style></head>
  <body><iframe src="{src}" title="Churn monitoring app" sandbox="allow-scripts allow-same-origin allow-forms allow-popups"></iframe></body>
</html>"""


def make_churn_dataset(random_state=42):
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=3000,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[0.75, 0.25],
        random_state=random_state,
    )

    rng = np.random.RandomState(random_state)

    def rescale(col, lo, hi):
        col_min, col_max = col.min(), col.max()
        return lo + (col - col_min) / (col_max - col_min) * (hi - lo)

    df = pd.DataFrame(
        {
            "customer_id": [f"CUST-{100000 + i}" for i in range(len(X))],
            "tenure_months": rescale(X[:, 0], 0, 72).round(0).astype(int),
            "monthly_charges": rescale(X[:, 1], 20, 120).round(2),
            "total_charges": rescale(X[:, 2], 50, 8000).round(2),
            "support_tickets": rescale(X[:, 3], 0, 12).round(0).astype(int),
            "avg_session_minutes": rescale(X[:, 4], 2, 90).round(1),
            "contract_type": pd.cut(
                X[:, 5],
                bins=3,
                labels=["month-to-month", "one-year", "two-year"],
            ).astype(str),
            "churned": y,
        }
    )
    # small amount of independent label noise so different n_estimators
    # values produce visibly different, but not wildly different, metrics
    flip = rng.rand(len(df)) < 0.03
    df.loc[flip, "churned"] = 1 - df.loc[flip, "churned"]
    return df


class ChurnFlow(ProjectFlow):
    n_estimators = Parameter("n-estimators", default=200, type=int)
    seed = Parameter("seed", default=42, type=int)
    card_app_url = Parameter(
        "card-app-url",
        default="http://localhost:5006/05_card_app",
        help="URL of the deployed Panel monitoring app embedded in the HTML Card",
    )

    @step_env(DEPS, python="3.13")
    @step
    def start(self):
        # Keep the evaluation cohort fixed across runs. The run parameter changes the model,
        # not the data, so the comparison app can support a valid candidate decision.
        self.df = make_churn_dataset(random_state=42)
        self.next(self.train)

    @step_env(DEPS, python="3.13")
    @step
    def train(self):
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder

        numeric_cols = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "support_tickets",
            "avg_session_minutes",
        ]
        categorical_cols = ["contract_type"]

        X = self.df[numeric_cols + categorical_cols]
        y = self.df["churned"]

        X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
            X, y, self.df["customer_id"], test_size=0.25, random_state=42
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", "passthrough", numeric_cols),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ]
        )
        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=self.n_estimators, random_state=self.seed
                    ),
                ),
            ]
        )
        pipeline.fit(X_train, y_train)

        self.pipeline = pipeline
        self.X_test = X_test
        self.y_test = y_test
        self.customer_id_test = idx_test
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.next(self.evaluate)

    @card(type="blank", id="summary")
    @card(type="html", id="interactive_monitoring")
    @step_env(EVAL_DEPS, python="3.13")
    @step
    def evaluate(self):
        import shap
        from metaflow.cards import Markdown, Table, ValueBox, VegaChart
        from sklearn.inspection import partial_dependence, permutation_importance
        from sklearn.metrics import (
            accuracy_score,
            confusion_matrix,
            f1_score,
            precision_recall_curve,
            precision_score,
            recall_score,
            roc_auc_score,
            roc_curve,
        )

        y_true = self.y_test.to_numpy()
        y_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        self.metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_proba)),
        }

        self.confusion = confusion_matrix(y_true, y_pred).tolist()

        fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
        self.roc_data = pd.DataFrame(
            {"fpr": fpr, "tpr": tpr, "threshold": np.append(roc_thresh, np.nan)[: len(fpr)]}
        )

        precision, recall, pr_thresh = precision_recall_curve(y_true, y_proba)
        self.pr_data = pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.append(pr_thresh, np.nan)[: len(precision)],
            }
        )

        feature_names = self.pipeline.named_steps["preprocess"].get_feature_names_out()
        importances = self.pipeline.named_steps["model"].feature_importances_
        self.feature_importance = pd.DataFrame(
            {"feature": feature_names, "importance": importances}
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        permutation = permutation_importance(
            self.pipeline,
            self.X_test,
            y_true,
            scoring="roc_auc",
            n_repeats=5,
            random_state=self.seed,
        )
        self.permutation_importance = pd.DataFrame(
            {
                "feature": self.X_test.columns,
                "importance": permutation.importances_mean,
                "standard_deviation": permutation.importances_std,
            }
        ).sort_values("importance", ascending=False).reset_index(drop=True)

        charges_dependence = partial_dependence(
            self.pipeline,
            self.X_test,
            features=["monthly_charges"],
            kind="average",
        )
        self.partial_dependence = pd.DataFrame(
            {
                "monthly_charges": charges_dependence["grid_values"][0],
                "average_churn_probability": charges_dependence["average"][0],
            }
        )

        transformed_test = self.pipeline.named_steps["preprocess"].transform(self.X_test)
        shap_values = shap.TreeExplainer(self.pipeline.named_steps["model"]).shap_values(
            transformed_test
        )
        if isinstance(shap_values, list):
            positive_class_values = shap_values[1]
        elif shap_values.ndim == 3:
            positive_class_values = shap_values[:, :, 1]
        else:
            positive_class_values = shap_values
        explanation_rows = []
        for customer_id, values in zip(self.customer_id_test, positive_class_values, strict=True):
            top_indices = np.argsort(np.abs(values))[-5:][::-1]
            explanation_rows.extend(
                {
                    "customer_id": customer_id,
                    "feature": feature_names[index],
                    "shap_value": float(values[index]),
                    "absolute_shap_value": float(abs(values[index])),
                }
                for index in top_indices
            )
        self.local_explanations = pd.DataFrame(explanation_rows)

        self.predictions = pd.DataFrame(
            {
                "customer_id": self.customer_id_test.to_numpy(),
                "contract_type": self.X_test["contract_type"].to_numpy(),
                "y_true": y_true,
                "y_pred": y_pred,
                "churn_probability": y_proba,
                "scored_week": pd.Timestamp("2026-01-05")
                + pd.to_timedelta(np.arange(len(y_true)) % 12, unit="W"),
            }
        )

        # The Card is a durable, run-scoped review summary. The Panel app later reads the
        # same persisted artifacts for exploration after the run completes.
        current.card["summary"].append(Markdown("# Churn model evaluation"))
        current.card["summary"].append(
            Table(
                [[
                    ValueBox("ROC AUC", f"{self.metrics['roc_auc']:.3f}"),
                    ValueBox("Precision", f"{self.metrics['precision']:.3f}"),
                    ValueBox("Recall", f"{self.metrics['recall']:.3f}"),
                    ValueBox("F1", f"{self.metrics['f1']:.3f}"),
                ]]
            )
        )
        current.card["summary"].append(
            Markdown(
                "## Explainability\n"
                "This run stores per-customer SHAP values, permutation importance, and a "
                "partial-dependence curve for the monitoring app."
            )
        )

        app_url = self.card_app_url
        if current.is_production and app_url.startswith("http://localhost"):
            from metaflow.apps import AppDeployer

            apps = AppDeployer.list_deployments(
                name="churn-monitoring-card-app",
                project=current.project_name,
                branch="main",
            )
            app_url = next((app.public_url for app in apps if app.public_url), None)
            if not app_url:
                raise RuntimeError("Deploy churn-monitoring-card-app before triggering the flow")
        app_src = f"{app_url}?flow={current.flow_name}&run_id={current.run_id}"
        if current.is_production:
            self.prj.register_model(
                "churn_classifier",
                "pipeline",
                description="Churn classifier trained and evaluated in this run",
                annotations={**self.metrics, "n_estimators": self.n_estimators, "card_app_url": app_url},
                tags={"framework": "scikit-learn", "demo": "session4"},
            )
        self.html = IFRAME_TEMPLATE.format(src=app_src)
        current.card["summary"].append(
            VegaChart(
                {
                    "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
                    "title": "Feature importance",
                    "data": {"values": self.feature_importance.head(8).to_dict("records")},
                    "mark": "bar",
                    "encoding": {
                        "x": {"field": "importance", "type": "quantitative"},
                        "y": {"field": "feature", "type": "nominal", "sort": "-x"},
                        "tooltip": [
                            {"field": "feature", "type": "nominal"},
                            {"field": "importance", "type": "quantitative", "format": ".3f"},
                        ],
                    },
                    "width": 500,
                    "height": 250,
                }
            )
        )

        self.next(self.end)

    @step_env(DEPS, python="3.13")
    @step
    def end(self):
        print(f"ChurnFlow run complete — n_estimators={self.n_estimators} metrics={self.metrics}")


if __name__ == "__main__":
    ChurnFlow()
