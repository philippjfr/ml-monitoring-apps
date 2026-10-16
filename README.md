# ML Monitoring Workshop Demos

Install the workshop environment once:

```bash
pixi install
```

Run the local apps with the Pixi tasks (the files in `artifacts/` are pre-generated and local-only, so they are not included in a fresh clone):

```bash
pixi run serve-wrapped          # Demo A: 01_wrapped.py
pixi run serve-monitoring       # Demo B: 02_interactive.py
pixi run serve-compare          # Demo C: 03_compare.py
pixi run seed-runs              # local ChurnFlow history for Demo D
pixi run serve-metaflow         # Demo D: 04_metaflow.py on local runs
pixi run serve-metaflow-remote  # Demo D: 04_metaflow.py on Outerbounds production runs
pixi run serve-card-app         # Demo G: 05_card_app.py
```

The Pixi environment pins Panel 1.9, which lacks `pn.ui`, so the notebook and serve tasks run in `dev3.14` through `conda run`. The Pixi activation still sets the local Metaflow datastore and `PYTHONPATH`.

`03_compare.py` and `04_metaflow.py` share the same UI and differ only in how they load runs. Both evaluate the decision rule at the weekly contact capacity (the top N customers by score): a candidate is promoted only if ROC AUC does not decline against the baseline, recall at capacity improves, and precision at capacity stays above the floor. A shared score threshold would penalise better-ranked models, which flag fewer customers at 0.5.

`05_card_app.py` reads the artifacts from the `ChurnFlow` run identified by its `flow` and
`run_id` query parameters. Without them, it opens the latest successful local run.
It uses `panel-material-ui` for the Page, sidebar selector, and layouts. Run it locally in
`dev3.14` (`pixi run serve-card-app` delegates to that environment).

## Outerbounds demo

The standalone Git project in `obproject.toml` deploys one `ChurnFlow` template and one Card app via `obproject-deploy`. The deployer reads the current Git branch and commit to build the project spec, which makes model assets written by production runs appear under the project's `main` branch. Commit the source and configuration on `main` before deploying; Git history alone is not enough to capture uncommitted files. From this directory, using an authenticated Outerbounds profile:

```bash
pixi install
pixi run deploy
pixi run trigger-pipeline
```

Deployment registers a template but does not run it. Trigger it at least once before opening the notebook. To compare models, trigger additional runs from the platform UI with different `n-estimators` values (for example 50 and 400). Open the `churn_workshop` project, select a `ChurnFlow` run, and inspect the graph, task artifacts, logs, and `evaluate` task's Cards. On production runs, `evaluate` registers the fitted pipeline as the `churn_classifier` model asset and publishes its iframe Card on the same step. The notebook covers EDA, Panel components and binding, then reads the pipeline, results, and run comparison from platform metadata; it does not train models or seed runs.

Run the notebook against Outerbounds with `pixi run notebook-remote`.

The EDA and Panel primer run without platform access; the results section needs a successful remote run. After switching from direct CLI deployment to `obproject-deploy`, deploy and trigger a **new run** to register the model on `main`. The earlier successful run wrote its model to `prod` and is not moved by redeployment.
The app-only update currently fails with an Outerbounds CLI `NoneType.rstrip` error. Git setup does not resolve that app deployment failure; check the Apps view and app readiness before presenting the iframe Card.

Local flow tasks (`pixi run flow` and `pixi run seed-runs`) use local metadata and datastore for offline demos. Use the service and S3 to inspect remote run history, not the local flow history.

### Card app

`deployments/card_app/config.yaml` deploys the Card app. It deliberately enables public access
and allows websocket origins because a run Card loads the app in an iframe. Confirm this access
model is acceptable for the target platform before deploying.

On production runs, the flow resolves `churn-monitoring-card-app` from the project's Apps tab automatically. The `card-app-url` parameter overrides this lookup when needed; the default `localhost` address is only used by local runs.

The evaluation task produces two Cards:

- `summary`: fixed metrics and global feature importance from the completed run.
- `interactive_monitoring`: an HTML Card that embeds `05_card_app.py` with the flow name and run ID.

The app provides prediction distributions, a synthetic weekly score trend, partial dependence,
permutation importance, and per-customer SHAP contributions. SHAP explains the model's
prediction, not the customer's real-world outcome. Compare runs only when their target
definition, evaluation cohort, and metric calculation are equivalent. `ChurnFlow` fixes its
synthetic dataset and test split across runs so `n_estimators` is the intended comparison.
