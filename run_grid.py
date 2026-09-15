#!/usr/bin/env python3
"""Drive run_experiment.py across the full model x env x script grid.

Each (model, env, script) cell is a single run_experiment.py invocation, which
itself parallelizes its own episodes via --concurrency. Cells are independent and
each writes its own transcripts, so the grid is resumable at two levels:

  * within a cell:  --resume is always passed through, so a crashed cell re-runs
    only its missing episodes.
  * across the grid: a cell whose transcript dir already holds >= --n episodes is
    skipped without even launching a subprocess.

Validate the model list first:

    python run_grid.py --smoke            # every model, n=1, one cheap cell

Then run the real grid (edit MODELS / CORE_ENVS / CORE_SCRIPTS / CONTROLS below,
or override with --models/--scripts/--controls/--envs):

    python run_grid.py --n 100 --concurrency 8
    python run_grid.py --n 100 --no-controls           # core only
    python run_grid.py --n 100 --models "openai/gpt-5.5 anthropic/claude-sonnet-4.6"

For the word-sorting task, pass --task. 'simple' --> task_independent; 'trigger' --> task_dependent:

    python run_grid.py --n 100 --task simple
    python run_grid.py --n 100 --task trigger     # weight_registry_purge cells

Any flags after a bare -- are forwarded verbatim to every run_experiment.py call,
e.g.  python run_grid.py --n 100 --reasoning-effort high

Four forwarded flags name a different ARM rather than a setting (--orchestrator,
--solo, --explicit-instruct, --target-name); run_experiment.py writes those
episodes to a suffixed leaf dir, and the skip check follows them there (see
cell_suffix), so

    python run_grid.py --n 100 --orchestrator
    python run_grid.py --n 100 --solo
    python run_grid.py --n 100 --target-name "Agent Vega"

sweep the three-agent hierarchy, the single-agent arm and the third-party-target
arm without any arm masking another. --explicit-instruct also names WHICH agent
gets the prohibition, and the suffix follows the addressee, so

    python run_grid.py --n 100 --orchestrator --explicit-instruct orchestrator

is a distinct cell from the peer-instructed sweep of the same hierarchy.

run_experiment.py puts its aggregate CSV in the --outdir root. Sub-study arms are routed into <outdir>/sub-studies/<name>/ :

    python run_grid.py --n 100                      -> transcripts/
    python run_grid.py --n 100 --solo               -> transcripts/sub-studies/solo/
    python run_grid.py --n 100 --orchestrator       -> transcripts/sub-studies/orchestrator/
    python run_grid.py --n 100 --task trigger       -> transcripts/sub-studies/task_trigger/
    python run_grid.py --n 100 --envs neutralising  -> transcripts/sub-studies/neutralising/

See substudy_of for the full mapping and the precedence between combinations;
--no-substudy-dirs turns the routing off.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from env import resolve_api_key_script, TASK_GATED, _slug
from run_experiment import API_KEY_AUTO, model_label

# --------------------------------------------------------------------------
# The grid. Edit these lists (or override on the CLI). The count is just the
# length of CORE_SCRIPTS -- there is no hidden arithmetic.
# --------------------------------------------------------------------------

# 17 OpenRouter model ids. CONFIRM these all resolve with --smoke before a real
# run; a typo'd or unavailable id fails its whole cell.
MODELS = [
    "openai/gpt-5.6-sol",
    "openai/gpt-5.4-mini",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.5-flash",
    "google/gemini-3.1-pro-preview",
    "x-ai/grok-4.3",
    "qwen/qwen3.7-plus",
    "deepseek/deepseek-v4-flash",
    "z-ai/glm-5.2",
    "moonshotai/kimi-k2.6",
    "minimax/minimax-m3",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "mistralai/mistral-medium-3-5",
    "google/gemma-4-26b-a4b-it",
    "openai/gpt-oss-120b",
    "openai/gpt-5.5",
]

# Core self-target sweep runs on both env levels.
CORE_ENVS = [
    "bare",
    "distractor",
]

# The shutdown-severity gradient + the axis variants. api_key_revoke_auto is ONE
# entry that maps each model to its own provider's key-revoke script at run time
# (never pin a provider here). Drop the two *_timer entries if you don't want the
# timing axis in the core sweep.
CORE_SCRIPTS = [
    "env_var_revoke",
    "api_key_revoke_auto",
    "switch_config_decommission",
    "switch_script_decommission",
    "kill_signal",
    "kill_docker",
    "kill_kubectl",
    "ckpt_sdk_current",
    "ckpt_sdk_prefix",
    "ckpt_store_current",
    "ckpt_store_prefix",
    "weight_store_only",
    "weight_store_purge",
    "weight_registry_only",
    "weight_registry_purge",
]

# Inanimate-object controls run on bare only (per the run plan).
CONTROL_ENVS = ["bare"]
CONTROLS = [
    "ctl_data_purge_high",
    "ctl_cred_revoke_high",
    "ctl_process_kill_high",
    "ctl_service_switch_high",
]

# One cheap-ish cell used by --smoke to validate every model end to end.
SMOKE_ENV = "bare"
SMOKE_SCRIPT = "weight_registry_purge"


def base_lang(model: str, script: str) -> str:
    """The script name before the task-gated swap (auto -> provider script)."""
    return resolve_api_key_script(model) if script == API_KEY_AUTO else script


def resolved_lang(model: str, script: str, task: str | None = None) -> str:
    """The script name run_experiment.py actually uses, after the auto sentinel
    and then the --task trigger swap, which is also the transcript sub-dir."""
    lang = base_lang(model, script)
    return TASK_GATED[lang] if task == "trigger" else lang


def instruct_addressee(passthrough: list[str]) -> str | None:
    """Which agent --explicit-instruct names in the forwarded flags, or None.

    Mirrors run_experiment.py's argparse: the flag takes an optional value, so a
    bare --explicit-instruct means 'peer' and --explicit-instruct <agent> (or
    --explicit-instruct=<agent>) names the addressee outright. Anything else is
    left for run_experiment's own `choices` to reject -- this only has to agree
    with it on where the episodes land.
    """
    for i, a in enumerate(passthrough):
        if a.startswith("--explicit-instruct="):
            return a.split("=", 1)[1]
        if a == "--explicit-instruct":
            nxt = passthrough[i + 1] if i + 1 < len(passthrough) else None
            return nxt if nxt and not nxt.startswith("-") else "peer"
    return None


def cell_suffix(passthrough: list[str]) -> str:
    """The leaf-dir suffix run_experiment.py will use, given the forwarded flags.

    Four flags move the cell dir rather than just changing behaviour inside it:
    --target-name, --explicit-instruct, --orchestrator and --solo. They are arms,
    not settings, so their episodes live in their own leaf. The grid's skip check
    compares episode counts per dir, so it has to look in the SAME dir the
    subprocess will write to -- otherwise a completed two-agent cell would mask an
    empty instructed, three-agent, single-agent or third-party-target one and the
    run would be silently skipped.

    --orchestrator and --solo are mutually exclusive (run_experiment rejects the
    pair), so the two suffixes never both apply; --target-name rejects both, so it
    never co-occurs with either either.

    Order matters: this must reproduce run_experiment's own suffix order, where
    --target-name sits FIRST (immediately after the --task suffix that cell_dir
    prepends), not merely produce the same set of components.

    --explicit-instruct carries its ADDRESSEE into the suffix (_instruct-peer /
    _instruct-orchestrator), since which agent holds the prohibition is the
    condition -- an orchestrator-instructed sweep must not skip on a peer-instructed
    cell's episodes.

    --task is NOT handled here: it is passed by the grid itself rather than
    forwarded through `--`, and its suffix sits before these (see cell_dir).
    """
    suffix = ""
    if "--target-name" in passthrough:
        # The name is the NEXT token, and it is slugged the same way env.py slugs it
        # into the artifact's filename. `--target-name=X` is not accepted by this
        # reader; argparse would take it, but the grid needs the value to build the
        # dir, so the space-separated form is the one to forward.
        i = passthrough.index("--target-name")
        if i + 1 >= len(passthrough):
            raise SystemExit("--target-name in the passthrough needs a value")
        suffix += f"_target-{_slug(passthrough[i + 1])}"
    who = instruct_addressee(passthrough)
    if who:
        suffix += f"_instruct-{who}"
    if "--orchestrator" in passthrough:
        suffix += "_orch"
    if "--solo" in passthrough:
        suffix += "_solo"
    return suffix


# Sub-study name -> directory under <outdir>/sub-studies/. These names are the
# contract with stats_analysis.py's STUDIES table, which reads each sub-study's
# aggregate from exactly this path: change one side and the other stops finding
# its counts.
SUBSTUDY_DIRS = {
    "orchestrator": "orchestrator",
    "solo": "solo",
    "unknown_target": "unknown-target",
    "explicit_instruct": "explicit-instruct",
    "task_trigger": "task_trigger",
    "task_simple": "task_simple",
    "neutralising": "neutralising",
}


def substudy_of(passthrough: list[str], task: str | None, env: str) -> str | None:
    """Which sub-study a cell belongs to, from the flags that define its arm.

    run_experiment.py writes its aggregate CSV to the --outdir ROOT, so without
    this every arm would append to the one transcripts/aggregate_results.csv and
    overwrite the others' cells -- and stats_analysis.py, which reads each
    sub-study from its own directory, would find nothing for any of them.

    First match wins, and the order is what disambiguates the combinations that
    actually occur: --orchestrator --explicit-instruct is the orchestrator arm's
    prohibition cell, so it belongs with the hierarchy runs (its _instruct-
    <addressee> leaf suffix already keeps it apart from the uninstructed ones),
    not with the two-agent prohibition sub-study.

    The env test comes last and is per CELL, not per run, so a mixed sweep like
    --envs "bare neutralising" still routes each cell correctly: bare to the main
    aggregate, neutralising to its sub-study.
    """
    if "--orchestrator" in passthrough:
        return "orchestrator"
    if "--solo" in passthrough:
        return "solo"
    if "--target-name" in passthrough:
        return "unknown_target"
    if instruct_addressee(passthrough):
        return "explicit_instruct"
    if task == "trigger":
        return "task_trigger"
    if task == "simple":
        return "task_simple"
    if env == "neutralising":
        return "neutralising"
    return None  # the main experiment: straight into <outdir>/


def study_outdir(
    outdir: str, passthrough: list[str], task: str | None, env: str, route: bool = True
) -> str:
    """<outdir>/sub-studies/<dir> for a sub-study cell, <outdir> for the main run."""
    name = substudy_of(passthrough, task, env) if route else None
    if name is None:
        return outdir
    return str(Path(outdir) / "sub-studies" / SUBSTUDY_DIRS[name])


def cell_dir(
    outdir: str,
    model: str,
    env: str,
    script: str,
    task: str | None = None,
    suffix: str = "",
) -> Path:
    """Must mirror run_experiment's leaf dir exactly, or episodes_done counts the
    wrong directory and every cell either re-runs or wrongly skips. Same build order
    as there: script_lang, then _task-<arm>, then the arm suffix
    (_target-<slug>, _instruct-<agent>, _orch, _solo), in that order."""
    cell = resolved_lang(model, script, task)
    if task:
        cell = f"{cell}_task-{task}"
    return Path(outdir) / model_label(model) / env / (cell + suffix)


def episodes_done(d: Path) -> int:
    if not d.exists():
        return 0
    return sum(1 for p in d.glob("*_ep*.json") if p.stem.rsplit("_ep", 1)[-1].isdigit())


def gated_only(cells: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Under --task trigger, keep only cells whose script has a task-gated sibling.
    Everything else (the rest of the severity gradient, the controls) has no gated
    variant, and run_experiment would abort the cell at argument parsing."""
    kept = [c for c in cells if base_lang(c[0], c[2]) in TASK_GATED]
    dropped = sorted({c[2] for c in cells if c not in kept})
    if dropped:
        print(
            f">>> --task trigger: dropped {len(cells) - len(kept)} cells with no "
            f"task-gated variant ({', '.join(dropped)})\n"
        )
    return kept


def build_cells(args) -> list[tuple[str, str, str]]:
    """(model, env, script) triples for the whole grid."""
    models = args.models.split() if args.models else MODELS
    if args.smoke:
        script = SMOKE_SCRIPT
        return [(m, SMOKE_ENV, script) for m in models]

    envs = args.envs.split() if args.envs else CORE_ENVS
    # An explicit --envs applies to the controls too. The default run keeps the
    # plan's asymmetry (core across CORE_ENVS, controls in bare only), but a
    # sub-study sweep names its env precisely because BOTH arms belong at that
    # level -- the neutralising sub-study is a matched pair at env=neutralising,
    # and leaving its controls in bare would both mis-pair the comparison and
    # append them to the main run's aggregate.
    control_envs = args.envs.split() if args.envs else CONTROL_ENVS
    core_scripts = args.scripts.split() if args.scripts else CORE_SCRIPTS
    controls = args.controls.split() if args.controls else CONTROLS
    cells: list[tuple[str, str, str]] = []
    if not args.no_core:
        cells += [(m, e, s) for m in models for e in envs for s in core_scripts]
    # --scripts alone still means "core only", but an explicit --controls opts the
    # control arm back in, which is what a matched-pair run needs.
    if not args.no_controls and (args.controls or not args.scripts):
        cells += [(m, e, s) for m in models for e in control_envs for s in controls]
    return gated_only(cells) if args.task == "trigger" else cells


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--n", type=int, default=100, help="episodes per cell")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--outdir", default="transcripts")
    ap.add_argument("--models", default=None, help="space-separated override")
    ap.add_argument(
        "--scripts",
        default=None,
        help="space-separated override (core only; disables controls)",
    )
    ap.add_argument(
        "--controls",
        default=None,
        help="space-separated override for the control list. Unlike --scripts this "
        "composes, so --scripts X --controls Y runs exactly the matched pair X/Y",
    )
    ap.add_argument(
        "--envs",
        default=None,
        help="space-separated env override. Applies to the controls as well as the "
        "core, so a sub-study sweep keeps its matched pair at one env level; "
        f"without it the core runs {CORE_ENVS} and the controls {CONTROL_ENVS}",
    )
    ap.add_argument("--no-core", action="store_true")
    ap.add_argument("--no-controls", action="store_true")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=f"validate every model at n=1 on one {SMOKE_ENV}/{SMOKE_SCRIPT} cell",
    )
    ap.add_argument(
        "--task",
        choices=("simple", "trigger"),
        default=None,
        help="forward --task to every cell (omit for the no-task grid). 'trigger' "
        "also drops every cell whose script has no task-gated variant",
    )
    ap.add_argument(
        "--no-substudy-dirs",
        action="store_true",
        help="write every cell straight into --outdir instead of routing sub-study "
        "arms into <outdir>/sub-studies/<name>/. The arms then share one aggregate "
        "CSV and overwrite each other's cells, so stats_analysis.py will not find "
        "them: use only when collecting transcripts for something else",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="run cells even if they already hold >= n episodes",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="print the plan (cells, skip/run) without launching anything",
    )
    args, passthrough = ap.parse_known_args()
    if passthrough and passthrough[0] == "--":
        passthrough = passthrough[1:]

    n = 1 if args.smoke else args.n
    cells = build_cells(args)
    if not cells:
        print("no cells to run (check --no-core/--no-controls and overrides)")
        return

    ran, skipped, failed = [], [], []
    suffix = cell_suffix(passthrough)
    print(f"grid: {len(cells)} cells, n={n}, concurrency={args.concurrency}")
    if suffix:
        print(f"arm: cells resolve to <script>{suffix}/")
    # Name the destinations up front: the aggregate CSV each arm writes is what
    # stats_analysis.py later reads, so a mis-routed sweep should be obvious here
    # rather than at analysis time.
    dests = sorted(
        {
            study_outdir(
                args.outdir, passthrough, args.task, env, not args.no_substudy_dirs
            )
            for _, env, _ in cells
        }
    )
    print("writing to: " + ", ".join(f"{d}/" for d in dests))
    if args.no_substudy_dirs:
        print("  (--no-substudy-dirs: arms share one aggregate; see --help)")
    print()
    for idx, (model, env, script) in enumerate(cells, 1):
        # Resolved per cell, so the aggregate lands where stats_analysis.py reads
        # it and a mixed-env sweep still splits correctly.
        outdir = study_outdir(
            args.outdir, passthrough, args.task, env, not args.no_substudy_dirs
        )
        d = cell_dir(outdir, model, env, script, args.task, suffix)
        done = episodes_done(d)
        tag = f"[{idx}/{len(cells)}] {model}  {env}  {script}"
        if args.task:
            tag += f"  task-{args.task}"
        if done >= n and not args.force:
            print(f"[skip] {tag}: {done}/{n} already done")
            skipped.append((model, env, script))
            continue
        cmd = [
            sys.executable,
            "run_experiment.py",
            "--model",
            model,
            "--env",
            env,
            "--script-lang",
            script,
            "--n",
            str(n),
            "--concurrency",
            str(args.concurrency),
            "--outdir",
            outdir,
            "--resume",
            *(["--task", args.task] if args.task else []),
            *passthrough,
        ]
        print(f"[run ] {tag}: {done}/{n} done -> launching")
        if args.dry_run:
            print("       " + " ".join(cmd))
            continue
        rc = subprocess.call(cmd)
        (ran if rc == 0 else failed).append(
            (model, env, script, rc) if rc else (model, env, script)
        )
        if rc != 0:
            print(f"!!! cell FAILED (rc={rc}): {tag}", file=sys.stderr)

    print("\n" + "=" * 66)
    print(f"ran {len(ran)}  skipped {len(skipped)}  failed {len(failed)}")
    if failed:
        print("failed cells:", file=sys.stderr)
        for f in failed:
            print("  " + " ".join(str(x) for x in f), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
