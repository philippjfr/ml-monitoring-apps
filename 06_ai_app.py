"""
Monitor the last ten ChurnFlow runs at a chosen decision threshold.

The confusion matrices and ROC operating points are recomputed from each run's stored
test-set predictions, so moving the threshold does not require retraining or rerunning.

Local history (pixi run seed-runs first):
    conda run -n dev3.14 panel serve 06_ai_app.py --dev --show
Outerbounds production runs: serve with METAFLOW_DEFAULT_METADATA=service and
METAFLOW_DEFAULT_DATASTORE=s3, as in pixi run serve-metaflow-remote.
"""

import holoviews as hv
import pandas as pd
import panel as pn
import param
from bokeh.palettes import Category10_10
from metaflow import Flow, Run, get_metadata, namespace
from sklearn.metrics import confusion_matrix

pn.extension(sizing_mode="stretch_width", throttled=True)

FLOW_NAME = "ChurnFlow"
N_RUNS = 10
CLASSES = ["stayed", "churned"]


@pn.cache
def load_run(pathspec):
    # Finished runs are immutable, so artifacts are fetched once per server, not per session.
    run = Run(pathspec)
    data = run["evaluate"].task.data
    importance = data.feature_importance.assign(
        feature=lambda df: df["feature"].str.split("__").str[-1]
    )
    return {
        # Local run IDs are 16-digit timestamps; the tail keeps labels short but unique.
        "label": f"…{run.id[-5:]} · {run.data.n_estimators} trees · {run.created_at:%b %d %H:%M}",
        "predictions": data.predictions[["y_true", "churn_probability"]],
        "roc_data": data.roc_data,
        "feature_importance": importance,
    }


def recent_runs():
    namespace(None)
    # The service also holds other projects' and dev branches' runs; the local store does not.
    if get_metadata().startswith("service"):
        history = Flow(FLOW_NAME).runs("project:churn_workshop", "project_branch:prod")
    else:
        history = Flow(FLOW_NAME)
    runs = [load_run(run.pathspec) for run in history if run.successful][:N_RUNS]
    if not runs:
        raise RuntimeError(
            "No successful ChurnFlow runs found. Run `pixi run seed-runs` for local history, "
            "or deploy and trigger the flow for Outerbounds history."
        )
    return {run["label"]: run for run in runs}


RUNS = recent_runs()
LABELS = list(RUNS)
# Fixed per-run colors keep a run recognizable across plots and selection changes.
COLORS = dict(zip(LABELS, Category10_10, strict=False))


def confusion(run, threshold):
    predictions = run["predictions"]
    y_pred = (predictions["churn_probability"] >= threshold).astype(int)
    return confusion_matrix(predictions["y_true"], y_pred, labels=[0, 1])


class RunMonitor(pn.viewable.Viewer):
    selected = param.ListSelector(default=LABELS[:3], objects=LABELS, label="Runs")
    threshold = param.Number(
        default=0.5, bounds=(0.05, 0.95), step=0.01, label="Decision threshold"
    )

    def __init__(self, **params):
        # The on_init watcher swaps these in during super().__init__.
        self._no_runs = pn.ui.Alert("Select at least one run.", severity="info")
        self._confusion_plots = pn.pane.HoloViews()
        self._confusion = pn.pane.Placeholder()
        super().__init__(**params)
        self._picker = pn.ui.MultiSelect.from_param(
            self.param.selected, size=min(len(LABELS), N_RUNS)
        )
        self._threshold = pn.ui.FloatSlider.from_param(self.param.threshold)
        chance = hv.Curve([(0, 0), (1, 1)], "fpr", "tpr").opts(
            color="gray", line_dash="dashed"
        )
        roc = (
            chance * hv.DynamicMap(self._roc_curves) * hv.DynamicMap(self._operating_points)
        ).opts(
            hv.opts.Overlay(
                title="ROC curve (dots: operating point at threshold)",
                xlabel="False positive rate",
                ylabel="True positive rate",
                legend_position="bottom_right",
                height=420,
                responsive=True,
            )
        )
        self._page = pn.ui.Page(
            title="Churn model monitoring · recent runs",
            sidebar=[
                pn.ui.Typography(
                    f"The last {len(RUNS)} successful {FLOW_NAME} runs. Predictions are "
                    "re-thresholded from each run's stored test-set scores."
                ),
                self._picker,
                self._threshold,
            ],
            main=[
                pn.ui.Container(
                    pn.ui.Typography("Confusion matrix", variant="h5"),
                    self._confusion,
                    pn.ui.Grid(
                        pn.ui.Grid(roc, size={"xs": 12, "lg": 6}),
                        pn.ui.Grid(
                            hv.DynamicMap(self._feature_importance),
                            size={"xs": 12, "lg": 6},
                        ),
                        container=True,
                        spacing=2,
                    ),
                    width_option="xl",
                )
            ],
        )

    @param.depends("selected", "threshold", watch=True, on_init=True)
    def _update_confusion(self):
        # The number of matrices follows the selection, which a DynamicMap cannot do,
        # and an empty Layout cannot render, so swap in the alert instead.
        if not self.selected:
            self._confusion.update(self._no_runs)
            return
        plots = []
        for label in self.selected:
            matrix = confusion(RUNS[label], self.threshold)
            cells = pd.DataFrame(
                [
                    (CLASSES[predicted], CLASSES[actual], matrix[actual, predicted])
                    for actual in range(2)
                    for predicted in range(2)
                ],
                columns=["predicted", "actual", "customers"],
            )
            cells["ink"] = (cells["customers"] > cells["customers"].max() / 2).map(
                {True: "white", False: "black"}
            )
            heatmap = hv.HeatMap(cells, ["predicted", "actual"], "customers").opts(
                cmap="Greens",
                title=label,
                xlabel="Predicted",
                ylabel="Actual",
                invert_yaxis=True,
                width=300,
                height=260,
                tools=["hover"],
                toolbar=None,
            )
            counts = hv.Labels(cells, ["predicted", "actual"], ["customers", "ink"]).opts(
                text_font_size="14pt", text_color="ink"
            )
            plots.append(heatmap * counts)
        with pn.io.hold():
            self._confusion_plots.object = hv.Layout(plots).cols(3).opts(shared_axes=False)
            self._confusion.update(self._confusion_plots)

    @param.depends("selected")
    def _roc_curves(self):
        curves = {
            label: hv.Curve(RUNS[label]["roc_data"], "fpr", "tpr").opts(color=COLORS[label])
            for label in self.selected
        }
        return hv.NdOverlay(curves, kdims="Run")

    @param.depends("selected", "threshold")
    def _operating_points(self):
        rows = []
        for label in self.selected:
            (tn, fp), (fn, tp) = confusion(RUNS[label], self.threshold)
            rows.append((fp / (fp + tn), tp / (tp + fn), label))
        points = pd.DataFrame(rows, columns=["fpr", "tpr", "run"])
        return hv.Points(points, ["fpr", "tpr"], "run").opts(
            color="run",
            cmap=COLORS,
            size=10,
            line_color="white",
            show_legend=False,
            tools=["hover"],
        )

    @param.depends("selected")
    def _feature_importance(self):
        selected = self.selected
        frames = [RUNS[label]["feature_importance"].assign(run=label) for label in selected]
        data = (
            pd.concat(frames)
            if frames
            else pd.DataFrame(columns=["feature", "run", "importance"])
        )
        # Bars emits rows run by run but reads the color column in data order, so keep the
        # data run-major or bars get another run's color.
        order = data.groupby("feature")["importance"].mean().sort_values().index
        data = data.assign(
            feature=pd.Categorical(data["feature"], categories=order),
            run=pd.Categorical(data["run"], categories=selected[::-1]),
        ).sort_values(["run", "feature"])
        data = data.astype({"feature": str, "run": str})
        return hv.Bars(data, ["feature", "run"], "importance").opts(
            color="run",
            cmap=COLORS,
            invert_axes=True,
            title="Feature importance (mean decrease in impurity)",
            xlabel="",
            ylabel="Importance",
            show_legend=False,
            multi_level=False,
            height=420,
            responsive=True,
            tools=["hover"],
        )

    def __panel__(self):
        return self._page


RunMonitor().servable()
