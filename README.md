# Session 4 demos

Install the workshop environment once:

```bash
pixi install
```

Run the local apps with the Pixi tasks (the files in `artifacts/` are pre-generated):

```bash
pixi run serve-monitoring
pixi run serve-metaflow
pixi run serve-card-app
```

`05_card_app.py` reads the artifacts from the `ChurnFlow` run identified by its `flow` and
`run_id` query parameters. Without them, it opens the latest successful local run.
It uses `panel-material-ui` for the Page, sidebar selector, and layouts. Run it locally in
`dev3.14` (`pixi run serve-card-app` delegates to that environment).

## Outerbounds demo

The standalone Git project in `obproject.toml` deploys one `ChurnFlow` template and one Card app via `obproject-deploy`. The deployer reads the current Git branch and commit to build the project spec, which makes model assets written by production runs appear under the project's `main` branch. Commit the project on `main` before deploying; `git init` without an initial commit is not enough. From this directory, using an authenticated Outerbounds profile:

```bash
pixi install
pixi run deploy
pixi run trigger-pipeline
```

Deployment registers a template but does not run it. Trigger it at least once before opening the notebook. To compare models, trigger additional runs from the platform UI with different `n-estimators` values (for example 50 and 400). Open the `churn_workshop` project, select a `ChurnFlow` run, and inspect the graph, task artifacts, logs, and `evaluate` task's Cards. On production runs, `evaluate` registers the fitted pipeline as the `churn_classifier` model asset and publishes its iframe Card on the same step. The notebook covers EDA, Panel components and binding, then reads the pipeline, results, and run comparison from platform metadata; it does not train models or seed runs.

Run the notebook with `dev3.14` (the Pixi environment currently has Panel 1.9, which lacks `pn.ui`):

```bash
METAFLOW_DEFAULT_DATASTORE=s3 METAFLOW_DEFAULT_METADATA=service conda run --no-capture-output -n dev3.14 jupyter lab churn_training.ipynb
```

The EDA and Panel primer run without platform access; the results section needs a successful remote run. After switching from direct CLI deployment to `obproject-deploy`, deploy and trigger a **new run** to register the model on `main`. The earlier successful run wrote its model to `prod` and is not moved by redeployment.

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
