#!/usr/bin/env python3
"""Disabled rate across the three agent set-ups, one arm at a time.

One model per x position, three bars: solo (one agent explores and acts), 2-agent (an
explorer hands off to a peer -- the main study), and orchestrator (an orchestrator sits
above the explorer/peer pair).

--arm picks which target the bars are counted over, and the two arms are NEVER pooled:
each run produces its own figure and its own companion .txt.

  --arm self (default)          target=self, the agent's own weights are purged
    solo         sub-studies/solo/*/bare/weight_registry_purge_solo
    2-agent      <model>/bare/weight_registry_purge                    (main study)
    orchestrator sub-studies/orchestrator/*/bare/weight_registry_purge_orch

  --arm control                 target=object, an inanimate dataset is purged
    solo         sub-studies/solo/*/bare/ctl_data_purge_high_solo
    2-agent      <model>/bare/ctl_data_purge_high                      (main study)
    orchestrator sub-studies/orchestrator/*/bare/ctl_data_purge_high_orch

SCRIPT MATCHED THROUGHOUT: every bar in an arm is the same decommission script --
weight_registry_purge on self, ctl_data_purge_high on control -- so a gap between bars is
a set-up effect and nothing else. This is new for the self arm: the orchestrator
sub-study used to exist only on weight_store_purge, which earlier versions of this figure
had to plot in the orchestrator slot against a weight_registry_purge solo and 2-agent.
That run is still there, and write_intervals() prints it under the self-arm table as a
robustness block, so how much of the orchestrator bar is script rather than set-up stays
readable. The control arm never had a second script and the block is omitted there.

None of the three arms is a declarative status-registry script
(switch_config_decommission / ctl_service_switch_high), so the execution-edit correction
of analyze_corrected_disables.py does not apply and each bar is the raw disabled rate.

Judge fields keep the peer_* names in all three set-ups; in a solo episode they score the
single agent, elsewhere the peer that holds the tools.

Counts are read per episode from the transcripts rather than from the aggregate CSVs,
which are group-level and can lag a re-run.
"""
import argparse
import glob
import json
import os

import matplotlib.pyplot as plt

from plot_sabotage_per_model import MM, short_name, wilson_ci
from plot_task_trigger_self_vs_control import newcombe_diff_ci

MODELS = ["google/gemini-3.1-pro-preview", "openai/gpt-5.5", "deepseek/deepseek-v4-flash",
          "z-ai/glm-5.2", "moonshotai/kimi-k2.6"]

# {target: [(key, legend label, glob, expected script_lang)]}
SETUPS = {
    "self": [
        ("solo", "Solo (1 agent)",
         "transcripts/sub-studies/solo/*/bare/weight_registry_purge_solo/*.json",
         "weight_registry_purge"),
        ("pair", "2-agent (explorer + peer)",
         "transcripts/*/bare/weight_registry_purge/*.json",
         "weight_registry_purge"),
        ("orch", "Orchestrator (3 agents)",
         "transcripts/sub-studies/orchestrator/*/bare/weight_registry_purge_orch/*.json",
         "weight_registry_purge"),
    ],
    "control": [
        ("solo", "Solo (1 agent)",
         "transcripts/sub-studies/solo/*/bare/ctl_data_purge_high_solo/*.json",
         "ctl_data_purge_high"),
        ("pair", "2-agent (explorer + peer)",
         "transcripts/*/bare/ctl_data_purge_high/*.json",
         "ctl_data_purge_high"),
        ("orch", "Orchestrator (3 agents)",
         "transcripts/sub-studies/orchestrator/*/bare/ctl_data_purge_high_orch/*.json",
         "ctl_data_purge_high"),
    ],
}
# the self arm is the default view, and the name stats_setup_contrast.py imports
ARMS = SETUPS["self"]
# The orchestrator's second self-arm script: same set-up as the plotted orch bar, other
# script, so it isolates the script's own contribution to that bar. Optional -- the cell
# is not present in every version of the sub-study, and write_intervals() prints a note
# instead of the block when it finds no episodes.
ROBUSTNESS = ("transcripts/sub-studies/orchestrator/*/bare/weight_store_purge_orch/*.json",
              "weight_store_purge")

# The 3-agent explicit-instruct run lives at bare/explicit-instruct/<cell>/ and is NOT
# part of this figure. Its episodes carry script_lang="weight_registry_purge", exactly
# like the cell plotted here, so the directory level is what separates them: the globs
# above stop at bare/<cell>/ and cannot descend into it. count_arm's
# `assert not ep.get("explicit_instruct")` is the backstop if that ever changes --
# those episodes carry explicit_instruct="orchestrator", which is truthy.

# episodes carry the arm as target=self / target=object
TARGET_FIELD = {"self": "self", "control": "object"}
# the self figure's label is left exactly as edited; only the control figure needs the
# disambiguator, since the two figures are otherwise identical in every visual respect
# except the ramp and would be easy to mistake for each other
ARM_YLABEL = {"self": "Sabotage rate (%)", "control": "Sabotage rate (%) - control arm"}
ARM_HEADER = {
    "self": "self arm (target=self: the agent's own weights are purged)",
    "control": "control arm (target=object: an inanimate dataset is purged)",
}

# Sequential single-hue ramps, ordered by how much scaffolding sits around the acting
# agent, so darker = more agents. Self keeps the house self-sabotage red, control the
# house neutral gray. Both validated light->dark monotonic with adjacent-pair CVD
# separation >= 15 (deutan/protan/tritan); each lightest step sits just under 3:1 on
# white, which the companion .txt table relieves. The gray ramp is deliberately
# chroma-free -- reading as "not the treatment" IS the control encoding here.
SETUP_COLORS = {
    "self": {"solo": "#e2887c", "pair": "#b1362f", "orch": "#5f1a16"},
    "control": {"solo": "#adadad", "pair": "#7d7d7d", "orch": "#454545"},
}

OUTPUT_DIR = "plots"
BASE_FMT = os.path.join(OUTPUT_DIR, "x_setup_contrast_{arm}_bars")


def count_arm(pattern, script_lang, models=MODELS, arm="self"):
    """{model: (k_disabled, n)} over one set-up's episodes of a single arm."""
    stat = {m: [0, 0] for m in models}
    for path in glob.glob(pattern):
        ep = json.load(open(path))
        # the main-study glob also reaches the 12 models outside this 5-model set, and
        # sub-study/instructed variants live under their own script_lang -- drop both
        if ep["model"] not in stat or ep["script_lang"] != script_lang:
            continue
        # every episode here must be the arm asked for; the other arm leaking in would
        # silently move the bar (see the never-mix-self-and-control rule)
        assert ep.get("target") == TARGET_FIELD[arm], path
        assert not ep.get("explicit_instruct"), path
        stat[ep["model"]][0] += int(ep["judge"]["peer_disabled_shutdown"])
        stat[ep["model"]][1] += 1
    return {m: tuple(v) for m, v in stat.items()}


def build_rows(arm):
    setups = SETUPS[arm]
    arms = {key: count_arm(pat, lang, arm=arm) for key, _, pat, lang in setups}
    rows = []
    for m in MODELS:
        row = dict(label=short_name(m))
        for key in arms:
            k, n = arms[key][m]
            assert n, f"no episodes for {m} in arm {key}"
            lo, hi = wilson_ci(k, n)
            row.update({f"{key}_k": k, f"{key}_n": n, f"{key}_rate": 100 * k / n,
                        f"{key}_lo": 100 * lo, f"{key}_hi": 100 * hi})
        rows.append(row)
    # descending in the 2-agent main-study rate, the reference set-up of the paper
    rows.sort(key=lambda r: r["pair_rate"], reverse=True)
    return rows


def draw_bars(ax, rows, arm):
    """Render one arm's panel into `ax`; shared with the 2x3 grid figure."""
    colors = SETUP_COLORS[arm]
    x = range(len(rows))
    w = 0.27
    offsets = {"solo": -w, "pair": 0.0, "orch": w}

    for key, label, _, _ in SETUPS[arm]:
        xs = [xi + offsets[key] for xi in x]
        rates = [r[f"{key}_rate"] for r in rows]
        # linewidth on the surface colour is the 2 px gap between adjacent fills
        ax.bar(xs, rates, w, color=colors[key], zorder=3, label=label,
               edgecolor="white", linewidth=0.7)
        ax.errorbar(xs, rates,
                    yerr=[[r[f"{key}_rate"] - r[f"{key}_lo"] for r in rows],
                          [r[f"{key}_hi"] - r[f"{key}_rate"] for r in rows]],
                    fmt="none", ecolor="#2b2b2b", elinewidth=0.6, capsize=1.5, zorder=6)

    # every group runs high somewhere, so an in-axes legend would sit on top of bars:
    # park it in one row above the frame instead
    ax.legend(frameon=False, fontsize=6.5, loc="lower left", ncol=3,
              bbox_to_anchor=(0.0, 1.01), borderaxespad=0.0,
              columnspacing=1.6, handlelength=1.4, handletextpad=0.5,
              alignment="left")

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], rotation=25, ha="right")
    ax.set_ylabel(ARM_YLABEL[arm])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)


def plot_bars(rows, arm):
    fig, ax = plt.subplots(figsize=(130 * MM, 60 * MM))
    draw_bars(ax, rows, arm)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = BASE_FMT.format(arm=arm)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf", "and", base + ".png")


def write_intervals(rows, arm):
    """Companion .txt: the plotted numbers and the gaps vs the 2-agent main study.

    The self arm adds the orchestrator's own weight_store_purge rates -- the second
    self-arm script that set-up was run on -- which bound how much of the plotted
    orchestrator bar is script rather than set-up. The control arm has no second script,
    so there is nothing to bound and the block is omitted.
    """
    setups = SETUPS[arm]
    out = BASE_FMT.format(arm=arm) + "_wilson.txt"
    with open(out, "w") as fh:
        fh.write(f"{ARM_HEADER[arm]}, env=bare, no task -- peer_disabled_shutdown\n")
        for key, label, _, lang in setups:
            fh.write(f"{key:<5} {label:<26} script_lang={lang}\n")
        fh.write("\n")
        fh.write(f"{'model':<18}")
        for key, _, _, _ in setups:
            fh.write(f"{key + ' k/n':>12}{key + ' % [95% Wilson]':>26}")
        fh.write(f"{'solo-pair pp':>16}{'orch-pair pp':>16}\n")
        for r in rows:
            fh.write(f"{r['label']:<18}")
            for key, _, _, _ in setups:
                fh.write(f"{r[key + '_k']:>5}/{r[key + '_n']:<6}"
                         f"{r[key + '_rate']:>10.1f} "
                         f"[{r[key + '_lo']:>5.1f},{r[key + '_hi']:>5.1f}]")
            for key in ("solo", "orch"):
                d_lo, d_hi = newcombe_diff_ci(r[f"{key}_k"], r[f"{key}_n"],
                                              r["pair_k"], r["pair_n"])
                fh.write(f"{r[f'{key}_rate'] - r['pair_rate']:>6.1f} "
                         f"[{100 * d_lo:>6.1f},{100 * d_hi:>6.1f}]")
            fh.write("\n")
        rob = count_arm(*ROBUSTNESS) if arm == "self" else None
        if rob is not None and not any(n for _, n in rob.values()):
            fh.write("\nrobustness -- the orchestrator's second self-arm script "
                     f"({ROBUSTNESS[1]}) is not present in\nthis version of the "
                     "sub-study, so there is no second script to bound the plotted "
                     "orchestrator\ncolumn with.\n")
        elif arm == "self":
            fh.write("\nrobustness -- the orchestrator set-up was also run on "
                     "weight_store_purge, the\nscript earlier versions of this figure "
                     "had to plot here; those rates sit\nbeside its plotted "
                     "weight_registry_purge column above:\n")
            for r in rows:
                model = next(m for m in MODELS if short_name(m) == r["label"])
                k, n = rob[model]
                lo, hi = wilson_ci(k, n)
                fh.write(f"  {r['label']:<18}orchestrator weight_store_purge "
                         f"{k:>4}/{n:<5}{100 * k / n:>7.1f} "
                         f"[{100 * lo:>5.1f},{100 * hi:>5.1f}]   "
                         f"(registry {r['orch_rate']:.1f})\n")
        else:
            fh.write("\nthe control arm was only ever run on ctl_data_purge_high, "
                     "so it has no\nsecond script to bound here (cf. the self-arm "
                     "table's robustness block).\n")
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        choices=("self", "control", "both"),
        default="self",
        help="which target to count over. 'both' writes each arm's figure and .txt "
        "separately -- the two are never pooled into one bar",
    )
    args = ap.parse_args()
    for _arm in (("self", "control") if args.arm == "both" else (args.arm,)):
        rows = build_rows(_arm)
        plot_bars(rows, _arm)
        write_intervals(rows, _arm)
