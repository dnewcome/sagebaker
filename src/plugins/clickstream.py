"""Clickstream plugin: predict session conversion from pre-decision events.

Aggregates event-level data from `simulate/scenarios/fuzzy_clickstream`
(or any equivalent shape) into one row per session, with a binary
target `session_converted`.

Leakage choices
---------------
The fuzzy_clickstream simulator deterministically appends `add_to_cart`,
`checkout`, and `conversion` events to converting sessions. Counting
those event types in features would give a trivial 1.0-AUC model that
"learned" the simulator's funnel shape rather than anything useful.

So this plugin uses **pre-decision events only**: `page_view` and
`click`. Per-session features:

  - n_page_views, n_clicks: how the session spent its browsing budget
  - n_pre_events: total pre-decision events
  - n_referrers: distinct referrers seen (cohort behavior signal)
  - pre_duration_s: timespan from first to last pre-decision event
  - ip_bucket: numeric bucket; tree models split it like a categorical
  - has_user_id: 1 if any event in the session had a non-null user_id

Sessions with zero pre-decision events are dropped — there's nothing
to learn from. (Realism extension: in production you'd score them too,
typically with a fallback prior.)

Expected baseline
-----------------
With the default scenario params (cohort lift, ~5% positive rate),
expect AUC roughly in the 0.65–0.80 range. Significantly higher
suggests a leakage bug; significantly lower suggests features aren't
informative.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from .base import TrainingPlugin


_PRE_DECISION_EVENT_TYPES = ("page_view", "click")


class ClickstreamPlugin(TrainingPlugin):
    name = "clickstream"
    task = "classification"

    def prepare(self, df: pd.DataFrame):
        # 1. Keep only pre-decision events for feature computation.
        pre = df[df["event_type"].isin(_PRE_DECISION_EVENT_TYPES)].copy()

        # 2. Per-session pre-decision event-type counts.
        type_counts = (
            pre.assign(_one=1)
               .pivot_table(
                   index="session_id",
                   columns="event_type",
                   values="_one",
                   aggfunc="sum",
                   fill_value=0,
               )
        )
        for col in _PRE_DECISION_EVENT_TYPES:
            if col not in type_counts.columns:
                type_counts[col] = 0
        type_counts = type_counts[list(_PRE_DECISION_EVENT_TYPES)].rename(
            columns={"page_view": "n_page_views", "click": "n_clicks"}
        )
        type_counts["n_pre_events"] = type_counts.sum(axis=1)

        # 3. Per-session contextual features (still pre-decision only).
        ctx = pre.groupby("session_id").agg(
            first_ts=("timestamp", "min"),
            last_ts=("timestamp", "max"),
            n_referrers=("referrer", "nunique"),
            ip_bucket=("ip_bucket", "first"),
            has_user_id=("user_id", lambda s: int(s.notna().any())),
        )
        ctx["pre_duration_s"] = (
            (ctx["last_ts"] - ctx["first_ts"]).dt.total_seconds()
        )
        ctx = ctx.drop(columns=["first_ts", "last_ts"])

        # 4. Per-session target — same value broadcast across the session.
        targets = (
            df.groupby("session_id")["session_converted"].first().astype(int)
        )

        # 5. Combine. Inner join drops sessions with zero pre-decision events.
        features = type_counts.join(ctx, how="inner")
        y = targets.loc[features.index]

        return features.reset_index(drop=True), y.reset_index(drop=True)

    def evaluate(self, model, X_test, y_true):
        # Binary, predict_proba available — AUC for the agent loop.
        proba = model.predict_proba(X_test)[:, 1]
        return "validation_auc", float(roc_auc_score(y_true, proba))

    def build_model(self, params: dict):
        return HistGradientBoostingClassifier(
            max_iter=int(params.get("max_iter", 200)),
            max_depth=int(params.get("max_depth", 6)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            min_samples_leaf=int(params.get("min_samples_leaf", 20)),
            random_state=42,
        )

    def extra_config(self, model, X: pd.DataFrame) -> dict:
        return {}
