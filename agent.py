"""autoresearch-style agent that iteratively improves a sagebaker plugin.

Inspired by karpathy/autoresearch — an LLM agent edits a small Python
file, runs training (~seconds on sonar-sized data), reads the metric,
keeps the change if better, reverts if worse, repeats. Cheap iteration
is the whole point.

Loop:
  1. Read agent_default.md — human-edited prompt with constraints + strategy
  2. Read the plugin file the agent is allowed to edit
  3. Ask Claude for a complete new version
  4. Write it, syntax-check, run `make train`, parse validation_accuracy
  5. Compare to the best so far; `git checkout --` to revert if worse
  6. Loop until --budget-seconds or --max-iterations

Prereqs:
  ANTHROPIC_API_KEY in .env (the Makefile auto-loads it into the kernel)
  pip install --group agent   # adds the anthropic SDK (or: make install-agent)
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
import shutil
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
    """Revert a file to its last committed state via git checkout.

    When the plugin lives in a different git repo (e.g. sagebaker-plugins),
    run git from that repo's directory so checkout finds the file.
    """
    repo_dir = str(Path(path).resolve().parent)
    subprocess.run(["git", "checkout", "--", str(Path(path).resolve())],
                   check=True, cwd=repo_dir)


KEEPS_DIR = Path(".agent-keeps")


def init_keeps_dir(plugin_stem):
    """Wipe and re-create this plugin's keeps subdir so 'best of this run'
    is unambiguous. Each agent run starts with a clean slate."""
    plugin_keeps = KEEPS_DIR / plugin_stem
    if plugin_keeps.exists():
        shutil.rmtree(plugin_keeps)
    plugin_keeps.mkdir(parents=True, exist_ok=True)
    return plugin_keeps


def save_keep(plugin_path, iter_n, metric, keeps_dir):
    """Snapshot the current plugin file to <keeps_dir>/iter-NNN-METRIC.py.

    The metric is encoded into the filename so `best_keep` can sort by it
    without parsing files. Filenames look like `iter-007-0.8219.py`.
    """
    fname = f"iter-{iter_n:03d}-{metric:.4f}.py"
    out = keeps_dir / fname
    out.write_text(Path(plugin_path).read_text())
    return out


def best_keep(keeps_dir):
    """Return the path of the highest-metric keep in keeps_dir, or None."""
    candidates = list(keeps_dir.glob("iter-*.py"))
    if not candidates:
        return None
    def metric_of(p):
        m = re.search(r"-(\d+\.\d+)\.py$", p.name)
        return float(m.group(1)) if m else 0.0
    return max(candidates, key=metric_of)


def revert_to_best(plugin_path, keeps_dir):
    """Restore the plugin file to the best-kept version of THIS run.

    Why this exists: `git checkout --` reverts to the last *committed*
    state, which is the pre-run baseline. If the agent kept a winning
    proposal at iter 7 (uncommitted) and then iter 12 fails, a naive
    git-checkout would blow away iter 7's work. Reverting to the
    best-keep file preserves wins instead.

    Falls back to git checkout if no keeps exist yet (we're early in
    the run and the baseline IS the best so far).
    """
    best = best_keep(keeps_dir)
    if best is not None:
        Path(plugin_path).write_text(best.read_text())
    else:
        revert(plugin_path)


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
No markdown fences, no diff format — just the file.
The FIRST line must be a comment in exactly this format:
# RATIONALE: <one sentence explaining what you changed and why>

    msg = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )
    src = strip_fences(msg.content[0].text)
    # Extract and strip the rationale line so it doesn't affect syntax or
    # byte-identity checks — but preserve it for MLflow tagging.
    rationale = ""
    lines = src.splitlines()
    if lines and lines[0].startswith("# RATIONALE:"):
        rationale = lines[0][len("# RATIONALE:"):].strip()
        src = "\n".join(lines[1:]).lstrip("\n")
    return src, rationale


def run_training(train_cmd, env, window=5):
    """Run training, show a rolling window of the last N output lines.

    Uses ANSI cursor-up to overwrite the window in place so the terminal
    doesn't scroll endlessly. All output is still captured for metric
    parsing. Merges stderr into stdout so warnings appear in order.
    """
    from collections import deque
    proc = subprocess.Popen(
        train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, env=env,
    )
    lines = []
    buf = deque(maxlen=window)
    printed = 0  # lines currently on screen; needed to know how far to move up
    for raw in proc.stdout:
        line = raw.rstrip("\n")
        lines.append(line)
        buf.append(line)
        if printed:
            sys.stdout.write(f"\033[{printed}A")  # move cursor up
        for dl in buf:
            sys.stdout.write(f"\r\033[K  | {dl}\n")  # clear line, write, newline
        printed = len(buf)
        sys.stdout.flush()
    proc.wait()
    return proc.returncode, "\n".join(lines)


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
    p.add_argument("--program", default="agent_default.md",
                   help="prompt with constraints + strategy hints")
    p.add_argument("--metric", default=None,
                   help="metric name to track (default: any validation_<name>=…)")
    p.add_argument("--max-iterations", type=int, default=50)
    p.add_argument("--budget-seconds", type=int, default=1800,
                   help="wall-clock cap (default 30 min)")
    p.add_argument("--data-dir", default="data",
                   help="dir containing the training CSV/parquet "
                        "(default: data; useful for running multiple agents "
                        "in parallel against different prepared datasets)")
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
        sys.exit("install agent deps first: pip install --group agent")

    program = read(args.program)
    client = Anthropic()

    # The --plugin flag is a file path (src/plugins/foo.py); the trainer
    # wants the bare plugin name ("foo"). Without this, agent.py would
    # edit foo.py while `make train` keeps running the *default* plugin —
    # which means the agent's edits have zero effect on what the trainer
    # actually trains. (Real bug we hit: agent on housing.py kept showing
    # sonar metrics because `make train` always uses DefaultPlugin.)
    plugin_name = Path(args.plugin).stem
    model_dir = os.environ.get("AGENT_MODEL_DIR", f"./models/{plugin_name}")
    train_script = os.environ.get("AGENT_TRAIN_SCRIPT", "src/train.py")
    train_cmd = [
        sys.executable, train_script,
        "--train", args.data_dir,
        "--model-dir", model_dir,
        "--plugin", plugin_name,
    ]
    print(f"plugin: {plugin_name} | trainer: {' '.join(train_cmd)}")

    # Per-run keeps dir: each successful proposal is snapshotted here so
    # later reverts can restore the best keep instead of the pre-run
    # baseline. Wiped at start of every run.
    keeps_dir = init_keeps_dir(plugin_name)
    print(f"keeps dir: {keeps_dir}")

    # ---- baseline run: run the *unmodified* plugin once, before involving
    # the LLM. If this fails, the data/plugin pair is broken and we bail
    # out cleanly with a useful message — no point letting the LLM try to
    # fix a bug that isn't its fault. The successful baseline metric
    # becomes `best`, so subsequent proposals only get kept if they
    # actually improve over the unmodified plugin.
    print("===== baseline run (unmodified plugin) =====")
    baseline_rc, baseline_out = run_training(train_cmd, os.environ.copy())
    if baseline_rc != 0:
        sys.exit(
            "baseline training failed BEFORE the agent loop started — your "
            "data/plugin pair is incompatible (often: ran `make data-movielens` "
            "then `make agent` which targets the DefaultPlugin expecting a "
            "supervised dataset with a `target` column).\n\n"
            "fix one of:\n"
            "  • re-prep with a compatible dataset (`make data-sonar`, "
            "`make data-iris`, `make data-housing`)\n"
            "  • point --plugin at one that matches the data\n"
        )
    baseline_metric = parse_metric(baseline_out, args.metric)
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
                proposal, rationale = propose(client, program, args.plugin, plugin_src,
                                              history, best, iters_since_improvement,
                                              diversify=args.diversify)
            except Exception as e:
                print(f"  LLM call failed: {e}")
                history.append(("", -1.0, False, f"LLM call failed: {e}"))
                iters_since_improvement += 1
                continue
            if rationale:
                print(f"  rationale: {rationale}")

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

            train_env = os.environ.copy()
            if rationale:
                train_env["AGENT_RATIONALE"] = rationale
            result_rc, result_out = run_training(train_cmd, train_env)
            if result_rc != 0:
                err_tail = result_out[-500:].strip()
                why = f"training failed (exit {result_rc}). last stdout:\n{err_tail}"
                print(f"  training failed (exit {result_rc}); reverting")
                revert_to_best(args.plugin, keeps_dir)
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            metric = parse_metric(result_out, args.metric)
            if metric is None:
                why = (f"training succeeded but no validation_<name>=… line in stdout "
                       f"(expected pattern '{args.metric or 'validation_<anything>'}')")
                print(f"  {why}; reverting")
                revert_to_best(args.plugin, keeps_dir)
                history.append((proposal, -1.0, False, why))
                iters_since_improvement += 1
                continue

            keep = metric > best
            print(f"  metric={metric:.4f} → {'KEEP' if keep else 'REVERT'}")
            if keep:
                best = metric
                iters_since_improvement = 0
                save_keep(args.plugin, i, metric, keeps_dir)
                history.append((proposal, metric, True, None))
            else:
                revert_to_best(args.plugin, keeps_dir)
                iters_since_improvement += 1
                why = (f"metric {metric:.4f} did not beat current best "
                       f"{best:.4f}")
                history.append((proposal, metric, False, why))
    except KeyboardInterrupt:
        interrupted = True
        print("\n\n  Ctrl-C — stopping the loop, restoring best keep.")

    # Final restore: ensure the on-disk plugin matches the best metric we
    # tracked. Without this, an interrupted run can leave the file in a
    # half-written proposal state.
    revert_to_best(args.plugin, keeps_dir)

    kept = sum(1 for entry in history if entry[2])
    header = "interrupted" if interrupted else "done"
    print(f"\n===== {header} =====")
    print(f"  iterations: {len(history)} ({kept} kept, {len(history) - kept} reverted)")
    if best > -float("inf"):
        print(f"  best metric: {best:.4f}")
        best_path = best_keep(keeps_dir)
        if best_path is not None:
            print(f"  restored final plugin from {best_path}")
        else:
            print(f"  no proposal beat baseline — plugin reverted to git HEAD")
    else:
        print("  no successful runs")
    print(f"  final plugin: {args.plugin}")


if __name__ == "__main__":
    main()
