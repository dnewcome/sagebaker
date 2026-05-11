"""Base class for task-specific training plugins.

Adding a new plugin
-------------------
1. Create ``src/plugins/<plugin>.py`` with a subclass of TrainingPlugin.
2. Register it in ``src/plugins/__init__.py``.

The generic harness (``src/train.py``) handles everything outside the plugin
contract: data loading, train/test split, bundle serialization, MLflow
tracking. The plugin owns feature engineering, the model class, the
hyperparameter defaults, and the validation metric.

Tasks
-----
A plugin declares its task via ``task = "classification"`` (default) or
``task = "regression"``. The default ``evaluate()`` method picks an
accuracy metric for classification; regression plugins override to return
R² (or any other higher-is-better metric of their choice).
"""
import pandas as pd


class TrainingPlugin:
    # Override in every subclass.
    name: str = "base"
    # Either "classification" or "regression". Drives default metric
    # choice and downstream dispatch in evaluate.py / local_serve.py.
    task: str = "classification"

    def prepare(self, df: pd.DataFrame) -> tuple:
        """Feature engineering + target extraction.

        Receives the raw DataFrame loaded from the train channel
        (CSV or parquet). Returns ``(X, y)`` where X is a DataFrame
        of model input features and y is a Series — integer labels
        for classification, continuous values for regression.

        Column pruning, type casting, and derived feature creation all
        belong here. Anything that must match at inference time should
        be mirrored in the plugin's corresponding inference helper.
        """
        raise NotImplementedError

    def evaluate(self, model, X_test, y_true) -> tuple:
        """Return ``(metric_name, value)`` for the held-out predictions.

        Higher-is-better convention — the harness, evaluate.py, agent.py
        and the MLflow registry promotion logic all assume larger means
        better. R² is the recommended regression metric (bounded above
        by 1, comparable across datasets); accuracy is the classification
        default.

        Receives the fitted model and the test features so subclasses
        can use ``predict_proba`` for continuous metrics like ROC-AUC
        (which gives the agent a much finer-grained signal than
        accuracy on small test sets).

        Override in subclasses for different metrics. The chosen name
        becomes the field name in metadata.json + the stdout log line
        (``validation_<name>=…``).
        """
        from sklearn.metrics import accuracy_score, r2_score
        y_pred = model.predict(X_test)
        if self.task == "regression":
            return "validation_r2", float(r2_score(y_true, y_pred))
        return "validation_accuracy", float(accuracy_score(y_true, y_pred))

    def build_model(self, params: dict):
        """Instantiate an unfitted sklearn-compatible estimator.

        ``params`` is a flat dict of strings — the same format SageMaker
        uses for hyperparameters.json. Parse what you need; ignore the
        rest. Provide defaults for anything that might not be present.

        Example::

            def build_model(self, params):
                return LGBMClassifier(
                    n_estimators=int(params.get("n_estimators", 100)),
                    num_leaves=int(params.get("num_leaves", 31)),
                )
        """
        raise NotImplementedError

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        """Extra fields to merge into config.json. Optional.

        Use this to record plugin-specific metadata that the inference
        side needs — e.g. which columns are categorical, the prediction
        threshold, or feature importance rankings.
        """
        return {}

    def load_bundle(self, model_dir: str):
        """Load the trained model from a bundle directory.

        Default: reads ``weights_file`` from config.json (falling back to
        ``model.joblib``) and deserializes with joblib. Override if the
        bundle layout differs — e.g. a custom pickle filename, a numpy
        archive, or a multi-file bundle.
        """
        import json
        import joblib
        import os
        weights_file = "model.joblib"
        config_path = os.path.join(model_dir, "config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                weights_file = json.load(f).get("weights_file", weights_file)
        return joblib.load(os.path.join(model_dir, weights_file))

    def serve(self, model, raw_input: list, config: dict) -> dict:
        """End-to-end inference for HTTP serving.

        Default implementation chains ``prepare_inference()`` →
        ``model.predict_proba()`` → ``postprocess()``. Override for models
        that don't use sklearn's predict_proba interface (recommenders,
        custom embedders, etc.).
        """
        X = self.prepare_inference(raw_input)
        raw = model.predict_proba(X)
        return self.postprocess(raw, config)

    def prepare_inference(self, raw_input: list) -> "pd.DataFrame":
        """Transform a raw HTTP request payload into model input features.

        ``raw_input`` is a list of dicts — the parsed JSON body from a POST
        request. Must produce the same column names and dtypes as ``prepare()``
        produces for X, so the same fitted model can be used for both.

        Raise ``NotImplementedError`` (the default) for plugins that are not
        yet wired to the serving harness.
        """
        raise NotImplementedError(
            f"Plugin '{self.name}' has no prepare_inference() implementation."
        )

    def postprocess(self, raw_output, config: dict) -> dict:
        """Format model output for the HTTP response.

        ``raw_output`` is whatever ``model.predict_proba()`` returns
        (shape n_samples × n_classes). ``config`` is the bundle's
        config.json dict — use it for threshold, label maps, etc.

        Default: binary classification with threshold from config.json
        (key ``prediction_threshold``, default 0.5). Override for
        recommenders, regression, multi-class output, etc.
        """
        import numpy as np
        threshold = float(config.get("prediction_threshold", 0.5))
        probs = raw_output[:, 1]
        predictions = (probs >= threshold).astype(int)
        return {"predictions": predictions.tolist()}

    def prepare_data(self, output_dir: str, seed: int = 42, extra_args: list = None) -> None:
        """Generate synthetic training data into ``output_dir``.

        Override in plugins that can generate their own synthetic data for
        local development and CI. The root-level ``prepare.py`` dispatches
        here so all prepare logic lives alongside the plugin it belongs to.

        Raise ``NotImplementedError`` (the default) if the plugin has no
        synthetic data generator — the caller will show a helpful message.
        """
        raise NotImplementedError(
            f"Plugin '{self.name}' has no prepare_data() implementation. "
            "Provide a real dataset via the --train argument instead."
        )
