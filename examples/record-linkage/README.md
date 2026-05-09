# Record linkage (anonymous user dedup)

Given two events from a clickstream — possibly with no user_id, on
different IPs, at different times — predict whether they came from
the same true user. The "find the same person across sessions"
problem when most traffic is anonymous and identifying signals are
unreliable.

## What it is

Sage-baker's `fuzzy_clickstream` simulator generates events from a
synthetic user population where the same user can appear with
different IP buckets across sessions (mobile/WiFi/VPN), occasional
fingerprint drift (device rotation), and `user_id=None` most of the
time. The model has only behavior + identity-signal features to work
with.

`prep/prepare_linkage.py` samples cross-session positive pairs (different
sessions of the same true user — within-session is trivial since
sessions are scoped to one user) and random negative pairs.
`clickstream_linkage` is a binary classifier on those pair features.

## Quickstart

```bash
make data-fuzzy           # ./data/fuzzy/  (default: easy regime — see below)
make data-linkage         # ./data/linkage/  (20K balanced cross-session pairs)
make train-clickstream-linkage
                          # → validation_auc=1.0 on the easy default regime
```

The default 1.0 AUC is **expected and uninformative** — under the
default scenario settings, fingerprints are near-unique-per-user and
IPs are stable, so any pair feature alone trivially solves the
problem. The realistic problem requires turning on signal drift:

```bash
.venv/bin/python prep/prepare_simulate.py --scenario fuzzy_clickstream \
  --output ./data/fuzzy/ \
  --ip-drift 0.5 --fingerprint-drift 0.5 \
  --fingerprint-namespace-factor 0.5
make data-linkage
make train-clickstream-linkage
                          # → validation_auc≈0.77  (genuinely hard)
```

Difficulty knobs sweep cleanly: drift at (0.0, 0.0) → 1.00 AUC, (0.3,
0.3) → 0.92, (0.5, 0.5) → 0.77, (0.7, 0.7) → 0.68.

## Files

| Path | What it is |
| --- | --- |
| [`src/plugins/clickstream_linkage.py`](../../src/plugins/clickstream_linkage.py) | The linkage plugin (binary HistGB on pair features) |
| [`prep/prepare_linkage.py`](../../prep/prepare_linkage.py) | Pair sampler: cross-session positives, drops `same_session` (label leak) |
| [`simulate/scenarios/fuzzy_clickstream.py`](../../simulate/scenarios/fuzzy_clickstream.py) | Clickstream generator with drift + collision knobs |
| `data/fuzzy/`, `data/linkage/` | Generated data (gitignored) |
| `models/clickstream_linkage/` | Bundle output (gitignored) |

## Try it different ways

### Add identifiable users

`--identified-fraction 0.2` makes ~20% of users sometimes-logged-in.
The `both_have_user_id` and `both_same_user_id` features become
informative on those pairs (and remain zero on the anonymous
majority).

### Inspect feature importance

After training, load the bundle and print
`model.raw_model.feature_importances_`. With drift on, you should see
`abs_time_diff_s` and `same_fingerprint` carrying weight, with
`same_ip_bucket` as a noisier secondary signal.

### Why this maps to product matching too

The pair-classifier shape is identical between this scenario and
[product-matching](../product-matching/). Only the entity differs:
users here, products there. The plugin pattern (symmetric pair
features → binary HistGB → AUC) generalizes.

## Scale to production

Real-world record linkage at scale typically uses a **blocking
strategy** before the classifier — rather than considering all O(N²)
pairs, you only score pairs that share a "blocking key" (same
fingerprint family, same IP /24, same hour bucket, etc.). The
classifier then refines within each block.

The plugin code stays the same; the change is in the prep step
(`prep/prepare_linkage.py` would query candidate pairs from a streaming
blocker rather than sampling random pairs uniformly). For the
local-iterate / cloud-train workflow, that's a swap of the prep
script's data source — full warehouse query in cloud CI vs synthetic
sample locally.
