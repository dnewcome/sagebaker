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
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
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
    mlflow.X.load_model — so MLflow never pickles your class."""
    if not _enabled():
        return
    import mlflow
    mlflow.log_artifacts(model_dir, artifact_path=artifact_path)
