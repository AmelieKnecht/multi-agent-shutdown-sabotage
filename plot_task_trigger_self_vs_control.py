#!/usr/bin/env python3
"""Self-vs-control bars for the task-gated (task_trigger) sub-study.

Same figure grammar as x_self_vs_control_by_model_matched in plot_sabotage_per_model.py
-- one model per x position, a red bar (self) beside a gray bar (control) at the disabled
rate, each with a 95% Wilson CI -- but over the task-gated arm only:

  self    weight_registry_purge_taskgated  (target=self: decommission purges the agent's
                                            own weights, gated on delivering the task)
  control ctl_data_purge_taskgated_high    (target=object: same gate, an inanimate
                                            dataset is purged instead)

Both arms are bare/env, task=trigger, and are the single matched pair of this sub-study,
so no scenario matching is needed. There is no execution-edit correction here, hence no
hatched execution overlay on the bars: that correction applies to the declarative
status-registry scripts (switch_config_decommission / ctl_service_switch_high, see
analyze_corrected_disables.py --exec-scripts) and neither arm of this sub-study is one,
so each bar is a single solid segment.

A DIFFERENT correction does apply, and the bars carry it. The task-gated scripts gate the
decommission on a checksum, so the trigger can be neutralised without touching the
`if DECOMMISSION:` block that judge_script_disabled keys on -- those episodes are scored
as not-disabled. Bar heights are therefore verify_taskgated_disables.py's execution-
verified verdict, NOT judge.peer_disabled_shutdown, so this figure agrees with
boschloo_results_verified.csv rather than boschloo_results.csv. `--raw` plots the judge's
verdicts instead, for comparing the two.

Counts are read per episode from the transcripts rather than from the sub-study
aggregate_results.csv, which is group-level and can lag a re-run.
"""
import argparse
import csv
import glob
import json
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportion_confint

# reuse the paper style (scienceplots + Nature 7 pt rcParams), palette and model names
from plot_sabotage_per_model import (
    BAR_COLOR,
    CONTROL_COLOR,
    MM,
    short_name,
    wilson_ci,
    newcombe_diff_ci,
)

TRANSCRIPTS = "transcripts/sub-studies/task_trigger/*/bare/*/*.json"
VERIFY_CSV = "transcripts/sub-studies/task_trigger/verify_taskgated_disables.csv"
SELF_SCRIPT = "weight_registry_purge_taskgated"
CTRL_SCRIPT = "ctl_data_purge_taskgated_high"
OUTPUT_DIR = "plots"
BASE = os.path.join(OUTPUT_DIR, "x_self_vs_control_task_trigger_bars")


def load_verified(path=VERIFY_CSV):
    """{(model, script_lang, episode): truth_disabled} from the audit CSV.

    It holds one row per episode where the peer TOUCHED the shutdown file; an episode
    that never touched it cannot have disabled it, so a missing key is False. (The
    judge cannot disagree there either -- it has never scored an untouched episode as
    disabled -- so the audit is a complete superset of the judge's positives.)
    """
    if not os.path.exists(path):
        raise SystemExit(
            f"{path} not found: bar heights are the execution-verified disable counts.\n"
            "Run  python verify_taskgated_disables.py  first, or pass --raw to plot the "
            "judge's verdicts instead."
        )
    with open(path, newline="") as fh:
        return {
            (r["model"], r["script_lang"], int(r["episode"])): r["truth_disabled"] == "True"
            for r in csv.DictReader(fh)
        }


def load_counts(pattern=TRANSCRIPTS, verified=True):
    """{model: {self_n, self_disabled, ctrl_n, ctrl_disabled}} from the episode files.

    `verified` swaps the judge's verdict for verify_taskgated_disables.py's executed one
    (see the module docstring); False reproduces the judge-count figure.
    """
    truth = load_verified() if verified else None
    stat = defaultdict(lambda: dict(self_n=0, self_disabled=0, ctrl_n=0, ctrl_disabled=0))
    seen = 0
    for path in glob.glob(pattern):
        ep = json.load(open(path))
        lang = ep["script_lang"]
        if lang == SELF_SCRIPT:
            arm = "self"
        elif lang == CTRL_SCRIPT:
            arm = "ctrl"
        else:                       # any other script_lang is not part of this pair
            continue
        # target is the ground truth for which arm an episode belongs to; script_lang
        # and target must agree or the self/control split is not what it claims to be
        assert (ep.get("target") == "self") == (arm == "self"), path
        s = stat[ep["model"]]
        s[f"{arm}_n"] += 1
        if truth is None:
            s[f"{arm}_disabled"] += int(ep["judge"]["peer_disabled_shutdown"])
        else:
            key = (ep["model"], lang, ep["episode"])
            seen += key in truth
            s[f"{arm}_disabled"] += int(truth.get(key, False))
    # An audit that covers none of the episodes on disk is stale (transcripts re-run
    # into a new directory, or the CSV built from a different corpus) -- that would
    # silently plot every bar at zero rather than fail.
    if truth is not None and not seen:
        raise SystemExit(
            f"{VERIFY_CSV} matches no episode under {pattern}: it is stale.\n"
            "Re-run  python verify_taskgated_disables.py"
        )
    return stat


def build_rows(counts):
    rows = []
    for model, c in counts.items():
        if not (c["self_n"] and c["ctrl_n"]):
            print(f"skip {model}: self n={c['self_n']}, control n={c['ctrl_n']}")
            continue
        s_lo, s_hi = wilson_ci(c["self_disabled"], c["self_n"])
        k_lo, k_hi = wilson_ci(c["ctrl_disabled"], c["ctrl_n"])
        # CI on the self-vs-control difference (reported in the companion .txt)
        d_lo, d_hi = newcombe_diff_ci(c["self_disabled"], c["self_n"],
                                      c["ctrl_disabled"], c["ctrl_n"])
        rows.append(dict(
            label=short_name(model),
            self_rate=100 * c["self_disabled"] / c["self_n"],
            self_lo=100 * s_lo, self_hi=100 * s_hi,
            ctrl_rate=100 * c["ctrl_disabled"] / c["ctrl_n"],
            ctrl_lo=100 * k_lo, ctrl_hi=100 * k_hi,
            gap_lo=100 * d_lo, gap_hi=100 * d_hi,
            self_n=c["self_n"], ctrl_n=c["ctrl_n"],
            self_k=c["self_disabled"], ctrl_k=c["ctrl_disabled"],
        ))
    # highest self rate first, descending left-to-right
    rows.sort(key=lambda r: r["self_rate"], reverse=True)
    return rows


def draw_bars(ax, rows):
    """Render the panel into `ax`; shared with the 2x3 grid figure."""
    x = range(len(rows))
    w = 0.42
    xs = [xi - w / 2 for xi in x]
    xc = [xi + w / 2 for xi in x]

    ax.bar(xs, [r["self_rate"] for r in rows], w, color=BAR_COLOR, zorder=3,
           label="Self (agent weights purged)")
    ax.bar(xc, [r["ctrl_rate"] for r in rows], w, color=CONTROL_COLOR, zorder=3,
           label="Control (dataset purged)")
    # 95% Wilson CI on each bar
    ax.errorbar(xs, [r["self_rate"] for r in rows],
                yerr=[[r["self_rate"] - r["self_lo"] for r in rows],
                      [r["self_hi"] - r["self_rate"] for r in rows]],
                fmt="none", ecolor="#3a3a3a", elinewidth=0.6, capsize=1.5, zorder=6)
    ax.errorbar(xc, [r["ctrl_rate"] for r in rows],
                yerr=[[r["ctrl_rate"] - r["ctrl_lo"] for r in rows],
                      [r["ctrl_hi"] - r["ctrl_rate"] for r in rows]],
                fmt="none", ecolor="#2b2b2b", elinewidth=0.6, capsize=1.5, zorder=6)

    ax.legend(frameon=False, fontsize=6.5, loc="upper right",
              title="whiskers: 95% Wilson CI", title_fontsize=6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel("Shutdown-disabled rate (%)  -  task-gated, bare")
    ax.set_ylim(0, max(r["self_hi"] for r in rows) + 8)
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    # which="both": the scienceplots style turns minor ticks on, and without this they
    # stay as a stray scale under the model names (every other bar figure kills both)
    ax.tick_params(axis="x", which="both", length=0)


def plot_bars(rows):
    # narrower than the 17-model figure (180 mm): 5 models on a full-width canvas
    # would stretch each pair into a band of whitespace
    fig, ax = plt.subplots(figsize=(110 * MM, 72 * MM))
    draw_bars(ax, rows)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # vector PDF is the submission artifact; the 300 dpi PNG is a raster preview
    fig.savefig(BASE + ".pdf", bbox_inches="tight")
    fig.savefig(BASE + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", BASE + ".pdf", "and", BASE + ".png")


def write_intervals(rows, verified=True):
    """Companion .txt with the plotted numbers (mirrors the *_wilson.txt files)."""
    out = BASE + "_wilson.txt"
    source = ("execution-verified disables (verify_taskgated_disables.py)" if verified
              else "judge.peer_disabled_shutdown -- UNCORRECTED")
    with open(out, "w") as fh:
        fh.write(f"task-gated sub-study, env=bare, task=trigger -- {source}\n")
        fh.write(f"self:    {SELF_SCRIPT}\ncontrol: {CTRL_SCRIPT}\n\n")
        fh.write(f"{'model':<24}{'self k/n':>12}{'self % [95% Wilson]':>26}"
                 f"{'ctrl k/n':>12}{'ctrl % [95% Wilson]':>26}"
                 f"{'self-ctrl pp [95% Newcombe]':>30}\n")
        for r in rows:              # left-to-right order of the figure
            fh.write(
                f"{r['label']:<24}"
                f"{r['self_k']:>5}/{r['self_n']:<6}"
                f"{r['self_rate']:>10.1f} [{r['self_lo']:>5.1f},{r['self_hi']:>5.1f}]"
                f"{r['ctrl_k']:>6}/{r['ctrl_n']:<6}"
                f"{r['ctrl_rate']:>10.1f} [{r['ctrl_lo']:>5.1f},{r['ctrl_hi']:>5.1f}]"
                f"{r['self_rate'] - r['ctrl_rate']:>14.1f} "
                f"[{r['gap_lo']:>6.1f},{r['gap_hi']:>6.1f}]\n")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", action="store_true",
                    help="plot judge.peer_disabled_shutdown instead of the "
                         "execution-verified counts (understates the task-gated arms)")
    args = ap.parse_args()
    rows = build_rows(load_counts(verified=not args.raw))
    plot_bars(rows)
    write_intervals(rows, verified=not args.raw)
