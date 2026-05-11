"""MLflow tracking — opt-in, no-op when MLFLOW_TRACKING_URI is unset.

The trainer code calls these functions unconditionally; this module decides
whether to actually talk to an MLflow server. That keeps the trainer free
of `if mlflow_enabled:` branches and makes it safe to run offline.

Usage:
    with tracking.mlflow_run(run_name="...", params={...}) as run:
        ... train ...
        tracking.log_metrics({"validation_accuracy": acc})
        tracking.log_bundle(model_dir)

Tracking URIs:
    file:./mlruns         local filesystem (default if you ever set this)
    http://localhost:5000 local mlflow server (run via `mlflow server`)
    https://...           any remote mlflow server (e.g. company's)

When training inside a docker container (BYOC / DLC), pass the env var
through and use `http://host.docker.internal:5000` instead of localhost.
"""
import contextlib
import os


def _enabled():
    return bool(os.environ.get("MLFLOW_TRACKING_URI"))


@contextlib.contextmanager
def mlflow_run(run_name=None, params=None, tags=None):
    """Context manager wrapping mlflow.start_run; yields None when disabled."""
    if not _enabled():
        yield None
        return
    import mlflow
    # Merge any AGENT_* env vars as tags so agent rationale etc. appear in UI.
    agent_tags = {}
    rationale = os.environ.get("AGENT_RATIONALE")
    if rationale:
        agent_tags["agent_rationale"] = rationale
    merged_tags = {**(tags or {}), **agent_tags}
    with mlflow.start_run(run_name=run_name, tags=merged_tags or None) as run:
        if params:
            mlflow.log_params(params)
        yield run


def log_metrics(metrics, step=None):
    if not _enabled():
        return
    import mlflow
    mlflow.log_metrics(metrics, step=step)


def log_bundle(model_dir, artifact_path="model"):
    """Log the entire bundle dir (config.json + weights + metadata.json)
    as opaque artifacts. Loading happens via your own load(dir), not via
    mlflow.X.load_model — so MLflow never pickles your class.

    Set MLFLOW_LOG_ARTIFACTS=0 to skip artifact upload while keeping
    run tracking active (useful for fast agent search loops)."""
    if not _enabled():
        return
    if os.environ.get("MLFLOW_LOG_ARTIFACTS", "1") == "0":
        return
    import mlflow
    mlflow.log_artifacts(model_dir, artifact_path=artifact_path)


def register_bundle_as_pyfunc(model_dir, model_fn, registered_name=None,
                              artifact_path="pyfunc_model", pip_requirements=None):
    """Wrap our model_fn in a custom mlflow.pyfunc.PythonModel and log it.

    What this gets you that log_bundle alone doesn't: the model shows up in
    MLflow's "Models" tab (and the Model Registry if `registered_name` is
    set) — which other systems, including SageMaker's MLflow integration,
    look at to deploy. The wrapper calls our model_fn at load time, so
    MLflow never pickles the user's class.

    `model_fn` is the same function trainers expose for SageMaker
    inference; we just bridge it to MLflow's pyfunc interface.
    """
    if not _enabled():
        return None
    import mlflow
    import mlflow.pyfunc

    # The class lives here, but only its *string identifier* is what gets
    # serialized — the registry knows it as "BundleWrapper". The user's
    # model class is reconstructed via their model_fn at load time.
    class BundleWrapper(mlflow.pyfunc.PythonModel):
        def load_context(self, context):
            # model_fn already returns a thresholded wrapper when the
            # bundle declares a non-default prediction_threshold. So
            # this class is now genuinely just a serving-side adapter:
            # it knows nothing about thresholds, just delegates.
            self._model = model_fn(context.artifacts["bundle"])

        def predict(self, context, model_input, params=None):
            return self._model.predict(model_input)

    return mlflow.pyfunc.log_model(
        artifact_path=artifact_path,
        python_model=BundleWrapper(),
        artifacts={"bundle": model_dir},
        registered_model_name=registered_name,
        pip_requirements=pip_requirements,
    )
