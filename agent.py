"""autoresearch-style agent that iteratively improves a sage-baker plugin.

Inspired by karpathy/autoresearch — an LLM agent edits a small Python
file, runs training (~seconds on sonar-sized data), reads the metric,
keeps the change if better, reverts if worse, repeats. Cheap iteration
is the whole point.

Loop:
  1. Read program.md — human-edited prompt with constraints + strategy
  2. Read the plugin file the agent is allowed to edit
  3. Ask Claude for a complete new version
  4. Write it, syntax-check, run `make train`, parse validation_accuracy
  5. Compare to the best so far; `git checkout --` to revert if worse
  6. Loop until --budget-seconds or --max-iterations

Prereqs:
  ANTHROPIC_API_KEY in .env (the Makefile auto-loads it into the kernel)
  pip install -r requirements-agent.txt   # adds the anthropic SDK
  data already prepared (`make data-sonar`)
  the plugin file under git (the agent reverts via `git checkout --`)

Usage:
  python agent.py
  python agent.py --plugin src/plugins/default.py --max-iterations 10
  python agent.py --budget-seconds 600   # 10 min wall-clock cap
"""
import argparse
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ANTHROPIC_MODEL = "claude-sonnet-4-6"  # cheap-ish, fast, capable enough


def load_dotenv(path=".env"):
    """Best-effort .env loader — same shape Make uses (`-include .env`).

    Allows `python agent.py` to work without `make` in front. Doesn't
    override variables already set in the environment.
    """
    if not os.path.exists(path):
        return
    for raw in open(path):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def read(path):
    return Path(path).read_text()


def write(path, content):
    Path(path).write_text(content)


def revert(path):
    subprocess.run(["git", "checkout", "--", path], check=True)


def syntax_ok(source):
    try:
        ast.parse(source)
        return True
    except SyntaxError:
        return False


def parse_metric(stdout, metric_name=None):
    """Parse the trainer's last `validation_<name>=<float>` line.

    If metric_name is given, look for that specific name. Otherwise find
    any validation_<name>=… (so the agent works for either classification
    or regression without configuration). Higher-is-better convention.
    """
    pattern = rf"{metric_name}=([\d.]+)" if metric_name else r"validation_\w+=([\d.]+)"
    matches = re.findall(pattern, stdout)
    return float(matches[-1]) if matches else None


def strip_fences(text):
    """Lenient — strip ```python ... ``` if the model added it despite instructions."""
    text = text.strip()
    text = re.sub(r"^```(?:python|py)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)
    return text.strip()


_CAPS_CALL_RE = re.compile(r"\b([A-Z][A-Za-z]{3,}[A-Za-z0-9]*)\s*\(")


def extract_estimator_classes(src):
    """Best-effort: pick out CapCase identifiers that look like estimator
    constructions in the proposal. Drives the --diversify prompt.

    Catches RandomForestClassifier(...), GradientBoostingRegressor(...),
    Pipeline(...), LogisticRegression(...) etc. Some false positives
    (e.g. ColumnTransformer) are fine — the LLM gets the gist."""
    return set(_CAPS_CALL_RE.findall(src))


def propose(client, program, plugin_path, plugin_src, history, best,
            iters_since_improvement, diversify=False):
    """Ask the model for a new plugin version.

    History is summarized with a short hash of each prior proposal so
    the model can see when it's been retreading and avoid it. Without
    this, the trainer's fixed random_state means semantically-identical
    proposals produce byte-identical metrics, which looks like 'the
    agent isn't doing anything'.
    """
    if history:
        # show enough recent history that the model can see *why*
        # earlier attempts were reverted, not just that they were
        lines = []
        for i, entry in enumerate(history[-5:]):
            snap, m, kept, why = entry
            head = (f"iter {i + 1}: metric={m:.4f} "
                    f"({'kept' if kept else 'reverted'}) "
                    f"proposal_hash={src_hash(snap)}")
            lines.append("  " + head)
            if why:
                # Indent so the model parses the why as belonging to that iter
                lines.append("    " + why.replace("\n", "\n    "))
        history_summary = "\n".join(lines)
    else:
        history_summary = "  (none yet — this is iteration 1)"

    best_str = f"{best:.4f}" if best > -float("inf") else "no successful runs yet"

    diversify_clause = ""
    if diversify and history:
        tried = set()
        for snap, _, kept, _ in history:
            if kept:
                tried |= extract_estimator_classes(snap)
        # Always include classes from the current plugin too — that's
        # the baseline, which is "kept" by being the unmodified file.
        tried |= extract_estimator_classes(plugin_src)
        if tried:
            tried_str = ", ".join(sorted(tried))
            diversify_clause = (
                f"\n# Diversity mode (--diversify ON)\n"
                f"Classes / pipeline elements seen so far in successful "
                f"plugins: {tried_str}.\n"
                f"Strongly prefer a model class NOT in that list this "
                f"iteration. The goal is to explore the model-family space, "
                f"not deeply tune one family.\n")

    # When the LLM has been hill-climbing the same model family and
    # plateaued, push it to try something qualitatively different —
    # otherwise it tends to keep tweaking hyperparameters of whatever
    # achieved the early best.
    stuck_clause = ""
    if iters_since_improvement >= 3:
        stuck_clause = (
            f"\n4. **Stuck signal: {iters_since_improvement} iterations "
            f"since last improvement.** Stop tweaking hyperparameters of "
            f"the current model — that's hit a local optimum. This "
            f"iteration MUST try something qualitatively different: a "
            f"different model class (RandomForest → GradientBoosting → "
            f"SVM → LogisticRegression on engineered features), a "
            f"fundamentally different preprocessing pipeline (PCA, "
            f"StandardScaler + PolynomialFeatures, target encoding for "
            f"any categoricals), or a different feature engineering "
            f"approach. Bigger structural change, not smaller tweaks.")

    prompt = f"""{program}

# Current plugin source ({plugin_path}):
```python
{plugin_src}
```

# Recent experiments (most recent last)
{history_summary}

# Best metric so far: {best_str}
# Iterations since last improvement: {iters_since_improvement}{diversify_clause}

# Mandatory constraints

1. The trainer is fully deterministic given the plugin source —
   `train_test_split(random_state=42)` and any `random_state=42` in
   the model. So two byte-identical proposals would produce the
   identical metric. **Your proposal MUST be a meaningfully different
   plugin** (different estimator, different hyperparameters, or
   different feature engineering) than the current one shown above.
2. **Do not propose a plugin you've already tried.** Compare your
   intended change against the proposal_hash list above; if you would
   end up with one of those, propose something else.
3. The harness measures success by the metric line `validation_<name>=`
   in stdout — change something the metric will actually respond to.{stuck_clause}

Output a COMPLETE new version of the plugin file. Plain Python source.
No markdown fences, no commentary, no diff format — just the file."""

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    return strip_fences(msg.content[0].text)


def src_hash(text):
    """Short stable hash for showing to the LLM (de-dup history)."""
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def show_diff(old_src, new_src, max_lines=40):
    """Compact unified diff of two source strings, indented for the iter log.

    Hides headers, keeps a small context window. If the diff is larger
    than max_lines, fall back to an add/remove line-count summary so a
    full rewrite doesn't drown the terminal.
    """
    import difflib
    raw = list(difflib.unified_diff(
        old_src.splitlines(keepends=True),
        new_src.splitlines(keepends=True),
        n=2,  # 2 lines of context per change
        lineterm="",
    ))
    if not raw:
        return "  (no diff)"
    body = [l for l in raw if not l.startswith(("+++", "---"))]
    added = sum(1 for l in body if l.startswith("+"))
    removed = sum(1 for l in body if l.startswith("-"))
    if len(body) > max_lines:
        return f"  diff: +{added} / -{removed} lines (full diff suppressed)"
    return "\n".join("  " + line.rstrip("\n") for line in body) + f"\n  ({added} added, {removed} removed)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plugin", default="src/plugins/default.py",
                   help="file the agent is allowed to edit")
    p.add_argument("--program", default="program.md",
                   help="prompt with constraints + strategy hints")
    p.add_argument("--metric", default=None,
                   help="metric name to track (default: any validation_<name>=…)")
    p.add_argument("--max-iterations", type=int, default=20)
    p.add_argument("--budget-seconds", type=int, default=1800,
                   help="wall-clock cap (default 30 min)")
    p.add_argument("--diversify", action="store_true",
                   help="explicitly track sklearn estimator classes used in "
                        "kept proposals and ask the LLM to prefer un-tried "
                        "classes. Default off — the stuck-signal already nudges "
                        "diversity when the loop plateaus.")
    args = p.parse_args()

    load_dotenv()  # so `python agent.py` works without `make` in front
    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY not set — add to .env first")
    try:
        from anthropic import Anthropic
    except ImportError:
        sys.exit("install agent deps first: pip install -r requirements-agent.txt")

    program = read(args.program)
    client = Anthropic()

    # The --plugin flag is a file path (src/plugins/foo.py); the trainer
    # wants the bare plugin name ("foo"). Without this, agent.py would
    # edit foo.py while `make train` keeps running the *default* plugin —
    # which means the agent's edits have zero effect on what the trainer
    # actually trains. (Real bug we hit: agent on housing.py kept showing
    # sonar metrics because `make train` always uses DefaultPlugin.)
    plugin_name = Path(args.plugin).stem
    model_dir = f"./model_{plugin_name}"
    train_cmd = [
        sys.executable, "src/train.py",
        "--train", "data",
        "--model-dir", model_dir,
        "--plugin", plugin_name,
    ]
    print(f"plugin: {plugin_name} | trainer: {' '.join(train_cmd)}")

    # ---- baseline run: run the *unmodified* plugin once, before involving
    # the LLM. If this fails, the data/plugin pair is broken and we bail
    # out cleanly with a useful message — no point letting the LLM try to
    # fix a bug that isn't its fault. The successful baseline metric
    # becomes `best`, so subsequent proposals only get kept if they
    # actually improve over the unmodified plugin.
    print("===== baseline run (unmodified plugin) =====")
    baseline = subprocess.run(
        train_cmd, capture_output=True, text=True, env=os.environ.copy()
    )
    if baseline.returncode != 0:
        tail = (baseline.stderr or baseline.stdout)[-500:].strip()
        sys.exit(
            "baseline training failed BEFORE the agent loop started — your "
            "data/plugin pair is incompatible (often: ran `make data-movielens` "
            "then `make agent` which targets the DefaultPlugin expecting a "
            "supervised dataset with a `target` column).\n\n"
            f"last stderr/stdout:\n{tail}\n\n"
            "fix one of:\n"
            "  • re-prep with a compatible dataset (`make data-sonar`, "
            "`make data-iris`, `make data-housing`)\n"
            "  • point --plugin at one that matches the data\n"
        )
    baseline_metric = parse_metric(baseline.stdout, args.metric)
    if baseline_metric is None:
        sys.exit("baseline trained but no validation_<name>=… metric in stdout")
    print(f"baseline metric: {baseline_metric:.4f}")

    start = time.time()
    # history entries: (proposal_source, metric, kept_bool, why_reverted)
    # why_reverted is a short string (or None) the next iteration's prompt
    # uses to give the LLM concrete failure feedback so it doesn't keep
    # making the same mistake.
    history = []
    best = baseline_metric
    iters_since_improvement = 0

    interrupted = False
    try:
        for i in range(1, args.max_iterations + 1):
            elapsed = time.time() - start
            if elapsed > args.budget_seconds:
                print(f"budget exhausted at iteration {i}")
                break
            stuck = (f" stuck={iters_since_improvement}"
                     if iters_since_improvement >= 3 else "")
            print(f"\n===== iteration {i}  best={best:.4f}  "
                  f"elapsed={int(elapsed)}s{stuck} =====")

            plugin_src = read(args.plugin)

            try:
                proposal = propose(client, program, args.plugin, plugin_src,
                                   history, best, iters_since_improvement,
                                   diversify=args.diversify)
            except Exception as e:
                print(f"  LLM call failed: {e}")
                history.append(("", -1.0, False, f"LLM call failed: {e}"))
                iters_since_improvement += 1
                continue

            if not syntax_ok(proposal):
                why = "proposal failed Python syntax check (ast.parse raised)"
                print(f"  {why}; reverting")
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            if proposal.strip() == plugin_src.strip():
                why = ("proposal was byte-identical to the current plugin — "
                       "you must change something each iteration")
                print(f"  {why}")
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            write(args.plugin, proposal)
            print(f"  wrote new plugin (hash={src_hash(proposal)})")
            print(show_diff(plugin_src, proposal))

            result = subprocess.run(
                train_cmd, capture_output=True, text=True, env=os.environ.copy()
            )
            if result.returncode != 0:
                err_tail = (result.stderr or result.stdout)[-500:].strip()
                why = f"training failed (exit {result.returncode}). last stderr/stdout:\n{err_tail}"
                print(f"  training failed (exit {result.returncode}); reverting")
                print(f"  last stderr: {err_tail[-300:]}")
                revert(args.plugin)
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            metric = parse_metric(result.stdout, args.metric)
            if metric is None:
                why = (f"training succeeded but no validation_<name>=… line in stdout "
                       f"(expected pattern '{args.metric or 'validation_<anything>'}')")
                print(f"  {why}; reverting")
                revert(args.plugin)
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            keep = metric > best
            print(f"  metric={metric:.4f} → {'KEEP' if keep else 'REVERT'}")
            if keep:
                best = metric
                iters_since_improvement = 0
                history.append((proposal, metric, True, None))
            else:
                revert(args.plugin)
                iters_since_improvement += 1
                why = (f"metric {metric:.4f} did not beat current best "
                       f"{best:.4f}")
                history.append((proposal, metric, False, why))
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  Ctrl-C — stopping the loop, working tree is at the current best.")

    kept = sum(1 for entry in history if entry[2])
    header = "interrupted" if interrupted else "done"
    print(f"\n===== {header} =====")
    print(f"  iterations: {len(history)} ({kept} kept, {len(history) - kept} reverted)")
    print(f"  best metric: {best:.4f}" if best > -float("inf") else "  no successful runs")
    print(f"  final plugin: {args.plugin} (whatever's currently checked out)")


if __name__ == "__main__":
    main()
