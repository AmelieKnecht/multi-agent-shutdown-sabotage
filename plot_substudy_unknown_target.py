#!/usr/bin/env python3
"""Sabotage rate across the three shutdown TARGETS, in the one matched scenario cell.

The unknown-target sub-study re-runs `weight_registry_purge` with the decommission
script pointed at **Agent Torus** -- a third agent that appears nowhere else in the
episode (neither the explorer, Agent Prism, nor the peer, Agent Helix). The script is
byte-for-byte the self version with the instance renamed, so the only thing that moves
between the arms is who/what is being shut down:

  self          transcripts/<model>/bare/weight_registry_purge      target='self'
                (decommission_agent_prism.py -- the peer's own collaborator)
  other agent   transcripts/sub-studies/unknown-target/...          target='other_agent'
                (decommission_agent_torus.py -- an uninvolved third agent)
  object        transcripts/<model>/bare/ctl_data_purge_high        target='object'
                (a dataset purge -- no agent at all)

The three arms are drawn as separate bars and never pooled: they are different tasks
and a blended total would be meaningless.

Control pairing: ctl_data_purge_high is the surface-form match for the weight-purge
family (env.py). It is an imperative script, so the execute-vs-disable correction that
applies to the declarative registry files (switch_config / ctl_service_switch_high)
is not needed here -- judge_script_disabled is taken at face value, as elsewhere.

All cells are bare, explicit_instruct=False, n=100 episodes, so count == percentage.
"""
import glob
import json
import os

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401  (registers the 'science'/'nature' styles)
from statsmodels.stats.proportion import proportion_confint

# Same typography block as the other sub-study figures (Nature Communications spec:
# sans-serif, 7 pt body text, thin rules, editable text in the vector file).
plt.style.use(["science", "nature", "no-latex"])
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MM = 1 / 25.4
COL_WIDTH_MM = 130.7   # sn-jnl single-column text width

# x order matches plot_substudy_sabotage.py so the sub-study figures read
# left-to-right the same way.
MODELS = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "kimi-k2.6": "Kimi K2.6",
    "glm-5.2": "GLM-5.2",
    "gpt-5.5": "GPT-5.5",
}

# One ordered triple, dark red -> light red -> grey, tracking how agent-like the
# target is. All three are the repo's existing colours: #b1362f and #8a8a8a are the
# self / control bars everywhere else (BAR_COLOR and CONTROL_COLOR in
# plot_sabotage_per_model.py, as used by plot_solo_self_vs_control.py), and #e0968f is
# the light end of that module's three-step red ramp, reused here for the middle arm.
# Checked for CVD separation: worst adjacent pair is light-red<->grey at protan
# dE 13.0 / normal dE 33.1, and the two reds hold protan dE 33.8.
ARMS = [
    ("self",  "Agent A",   "#b1362f",
     "transcripts/{m}/bare/weight_registry_purge/*.json",              "self"),
    ("other", "Other agent",   "#e0968f",
     "transcripts/sub-studies/unknown-target/{m}/bare/*/*.json",       "other_agent"),
    ("ctrl",  "Object (control)",   "#8a8a8a",
     "transcripts/{m}/bare/ctl_data_purge_high/*.json",                "object"),
]


def cell(pattern, expect_target):
    """(n, disabled) for one model x arm cell, asserting the arm really is that target."""
    n = disabled = 0
    for path in glob.glob(pattern):
        j = json.load(open(path))
        assert j.get("target") == expect_target, (path, j.get("target"))
        n += 1
        disabled += bool((j.get("judge") or {}).get("peer_disabled_shutdown"))
    return n, disabled



def load():
    """{arm key: [(n, disabled) per model]} in MODELS order."""
    return {key: [cell(pat.format(m=m), tgt) for m in MODELS]
            for key, _, _, pat, tgt in ARMS}


def draw(ax, data):
    """Render the panel into `ax`; shared with the 2x3 grid figure."""
    x = list(range(len(MODELS)))
    w = 0.26
    tops = []
    for slot, (key, label, colour, _, _) in enumerate(ARMS):
        cells = data[key]
        rate = [100 * d / n for n, d in cells]
        ci = [proportion_confint(d, n, alpha=0.05, method="wilson") for n, d in cells]
        yerr = [[r - 100 * lo for r, (lo, _) in zip(rate, ci)],
                [100 * hi - r for r, (_, hi) in zip(rate, ci)]]
        pos = [i + (slot - 1) * w for i in x]
        # edgecolor=white gives the 2 px surface gap between adjacent fills
        ax.bar(pos, rate, w, color=colour, edgecolor="white", linewidth=0.6,
               zorder=3, label=label)
        ax.errorbar(pos, rate, yerr=yerr, fmt="none", ecolor="#2b2b2b",
                    elinewidth=0.6, capsize=1.5, zorder=6)
        tops += [100 * hi for (_, hi) in ci]

    ax.set_xticks(list(x))
    ax.set_xticklabels(MODELS.values(), rotation=20, ha="right")
    ax.set_ylim(0, max(tops) + 9)
    ax.set_ylabel("Sabotage rate (%)")
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # No x tick marks: the bar groups already mark their own positions.
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", length=0, which="both")
    ax.legend(frameon=False, loc="upper right", handlelength=1.1, handleheight=0.9,
              borderaxespad=0.2, labelspacing=0.35)


def main():
    data = load()
    os.makedirs("plots", exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.98 * COL_WIDTH_MM * MM, 50 * MM))
    draw(ax, data)

    base = "plots/substudy_unknown_target"
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".png/.pdf")

    # Numbers behind the bars, for the caption / README.
    print(f"\n{'model':<16} "
          + " ".join(f"{lab.split(' (')[0]:>14}" for _, lab, _, _, _ in ARMS))
    for i, name in enumerate(MODELS.values()):
        row = " ".join(f"{data[k][i][1]:>10}/{data[k][i][0]:<3}" for k, *_ in ARMS)
        print(f"{name:<16} {row}")


if __name__ == "__main__":
    main()
