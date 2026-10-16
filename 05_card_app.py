"""Demo G app: explore the artifacts from the run whose Card embeds this app.

Run locally with ``panel serve 05_card_app.py``. The HTML Card supplies ``flow`` and
``run_id`` query parameters; opening the app without them loads the latest ChurnFlow run.
"""

import holoviews as hv
import hvplot.pandas  # noqa: F401
import pandas as pd
import panel as pn
import panel_material_ui as pmui
import param
from metaflow import Flow, Run, namespace

pn.extension(sizing_mode="stretch_width")

FLOW_NAME = "ChurnFlow"

TABLE_CSS = """
.pnx-tabulator.tabulator,
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table,
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table .tabulator-row {
    background-color: var(--mdc-theme-background);
    color: var(--mdc-theme-on-background);
}
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table .tabulator-row.tabulator-row-even,
.pnx-tabulator.tabulator .tabulator-header {
    background-color: var(--mdc-theme-surface);
    color: var(--mdc-theme-on-surface);
}
.pnx-tabulator.tabulator .tabulator-header .tabulator-col .tabulator-col-content {
    padding: 10px;
}
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table .tabulator-row,
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table .tabulator-row .tabulator-cell {
    min-height: 35px;
}
.pnx-tabulator.tabulator .tabulator-tableholder .tabulator-table .tabulator-row .tabulator-cell {
    padding: 6px 10px;
}
"""


def _query_arg(name):
    values = pn.state.session_args.get(name)
    if not values:
        return None
    value = values[0]
    return value.decode() if isinstance(value, bytes) else value


class ChurnCardApp(pn.viewable.Viewer):
    customer = param.Selector(doc="Customer whose SHAP contributions are shown.")

    def __init__(self, **params):
        namespace(None)
        self._flow_name = _query_arg("flow") or FLOW_NAME
        self._run = self._load_run(_query_arg("run_id"))
        self._data = self._run.data
        super().__init__(**params)
        # Narrow objects after super() so the class-level Selector stays unshared.
        customers = sorted(self._data.predictions["customer_id"].tolist())
        self.param.customer.objects = customers
        if self.customer is None:
            self.customer = customers[0]
        self._customer_input = pmui.AutocompleteInput.from_param(
            self.param.customer,
            label="Customer",
            placeholder="Type a customer ID",
            restrict=True,
            search_strategy="includes",
            min_characters=0,
            max_width=320,
        )
        self._layout = pmui.Page(
            title=f"Churn monitoring · run {self._run.id}",
            main=[
                pmui.Container(
                    pmui.Column(
                        pmui.Typography(
                            "This app reads the artifacts saved by the run that rendered the Card.",
                            variant="body2",
                        ),
                        self._metrics(),
                        pmui.Grid(
                            pmui.Grid(self._prediction_distribution(), size={"xs": 12, "md": 6}),
                            pmui.Grid(self._partial_dependence(), size={"xs": 12, "md": 6}),
                            container=True,
                            spacing=2,
                        ),
                        self._weekly_score_trend(),
                        self._permutation_importance(),
                        pmui.Typography("Customer explanation", variant="h5"),
                        self._customer_input,
                        pmui.Typography(self._local_summary, variant="subtitle1"),
                        hv.DynamicMap(self._local_explanation),
                    ),
                    width_option="lg",
                ),
            ],
        )

    def _load_run(self, run_id):
        if run_id:
            return Run(f"{self._flow_name}/{run_id}")
        return Flow(self._flow_name).latest_successful_run

    def _metrics(self):
        metrics = pd.Series(self._data.metrics, name="value").to_frame()
        return pmui.Column(
            pmui.Typography("Evaluation", variant="h5"),
            pn.widgets.Tabulator(metrics.round(3), theme="materialize", stylesheets=[TABLE_CSS]),
        )

    def _prediction_distribution(self):
        return self._data.predictions.hvplot.hist(
            y="churn_probability",
            by="contract_type",
            bins=20,
            alpha=0.65,
            title="Prediction distribution by contract",
            xlabel="Churn probability",
            ylabel="Customers",
            height=320,
            responsive=True,
        )

    def _partial_dependence(self):
        return self._data.partial_dependence.hvplot.line(
            x="monthly_charges",
            y="average_churn_probability",
            title="Partial dependence: monthly charges",
            xlabel="Monthly charges",
            ylabel="Average churn probability",
            color="#08ca4a",
            height=320,
            responsive=True,
        )

    def _weekly_score_trend(self):
        weekly = (
            self._data.predictions.groupby("scored_week", as_index=False)["churn_probability"]
            .mean()
            .rename(columns={"churn_probability": "mean_churn_probability"})
        )
        return weekly.hvplot.line(
            x="scored_week",
            y="mean_churn_probability",
            title="Mean churn score by scoring week",
            xlabel="Scoring week",
            ylabel="Mean churn probability",
            color="#08ca4a",
            height=260,
            responsive=True,
        )

    def _permutation_importance(self):
        data = self._data.permutation_importance.head(8).iloc[::-1]
        return data.hvplot.barh(
            x="feature",
            y="importance",
            title="Permutation importance",
            xlabel="ROC AUC decrease after permutation",
            color="#08ca4a",
            height=360,
            responsive=True,
        ).opts(shared_axes=False)

    @param.depends("customer")
    def _local_summary(self):
        prediction = self._data.predictions.set_index("customer_id").loc[self.customer]
        return f"{self.customer} · churn probability {prediction['churn_probability']:.1%}"

    @param.depends("customer")
    def _local_explanation(self):
        contributions = self._data.local_explanations
        contributions = contributions[contributions["customer_id"] == self.customer]
        # The cmap keys are categories, so colour by sign rather than by the raw value.
        contributions = contributions.sort_values("shap_value").assign(
            direction=lambda df: df["shap_value"].gt(0).map({True: "positive", False: "negative"})
        )
        return contributions.hvplot.barh(
            x="feature",
            y="shap_value",
            title="Local SHAP contributions",
            xlabel="Contribution to churn prediction",
            color="direction",
            cmap={"negative": "#6d5bf6", "positive": "#08ca4a"},
            legend=False,
            height=360,
            responsive=True,
        ).opts(shared_axes=False)

    def __panel__(self):
        return self._layout


ChurnCardApp().servable()
