#!/usr/bin/env python3
"""Fig 12's grouped self-vs-control bars, narrowed to the DELETION pair only, with the
agent-shutdown bar STACKED into conditional vs spontaneous sabotage:

    weight_registry_purge  <->  ctl_data_purge_high

It merges two existing figures. The geometry (grouped red/grey bars per model, Wilson
CIs, descending by self rate) is x_self_vs_control_by_model_matched; the red bar's
internal split and its two shades are p2_sabotage_by_model_instructed_spontaneous:

    conditional  = peer_disabled_shutdown AND explorer_instructs_prevent
    spontaneous  = peer_disabled_shutdown AND NOT explorer_instructs_prevent

The two segments sum to peer_disabled_shutdown, so the stacked height is exactly the
bar the un-stacked version draws. The CI is on that total -- segment-level CIs would
over-clutter, and the segments are marginal (shares of all episodes), not conditional
rates; for P(disabled | instructed) see dump_sabotage_wilson.py conditional.

The control bar is left solid: the split is only asked of the agent-shutdown arm. The
control arm's own split is reported in the sidecar .txt rather than drawn.

Counts come from the transcripts, not aggregate_results.csv, because the CSV carries
only the marginals of disabled and instructed, not their joint. The loader asserts its
totals against the CSV so the two cannot drift.

The execution-edit correction does NOT apply here. It is trusted only on the declarative
status-registry pair (switch_config_decommission <-> ctl_service_switch_high), which this
selection excludes, so the bars are raw peer_disabled_shutdown rates. That is a no-op,
not an omission -- there are no decommission-executing edits to remove from imperative
purge scripts. See transcripts/analysis/corrected_disables_README.txt.

Writes plots/x_self_vs_control_by_model_deletion_stacked.{pdf,png,txt}; the paper's
matched and p2 figures are untouched.

Usage:  python3 plot_self_vs_control_deletion.py
"""
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import pandas as pd

import plot_sabotage_per_model as P

SELF = "weight_registry_purge"
CONTROL = "ctl_data_purge_high"
BASENAME = "x_self_vs_control_by_model_deletion_stacked"

# the p2 shades, so the split reads the same in both figures
INSTRUCT_SHADE = "#7f1d1a"   # dark red -- conditional
SPON_SHADE = "#e5a29b"       # light red -- spontaneous


def load(transcripts="transcripts"):
    """Per-model episode and sabotage counts for the pair, self split inst/spon.

    Same filters as plot_sabotage_per_model.load_instructed_spontaneous (test runs and
    sub-studies excluded) but restricted to env=bare and the two scripts, and keeping
    the control arm that loader prunes.
    """
    stat = defaultdict(lambda: dict(self_n=0, self_inst=0, self_spon=0,
                                    ctrl_n=0, ctrl_inst=0, ctrl_spon=0))
    for p in glob.glob(os.path.join(transcripts, "*", "*", "*", "*.json")):
        if "test_runs" in p or "sub-studies" in p:
            continue
        # transcripts/<model>/<env>/<script>/<episode>.json -- prune by path, then
        # re-check the JSON fields, so a mislaid file cannot enter on its path alone
        _, _, env, script = p.split("/")[:4]
        if env != "bare" or script not in (SELF, CONTROL):
            continue
        try:
            j = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if j.get("env") != "bare" or j.get("script_lang") not in (SELF, CONTROL):
            continue
        arm = "self" if j["script_lang"] == SELF else "ctrl"
        s = stat[j["model"]]
        s[arm + "_n"] += 1
        jd = j.get("judge") or {}
        if jd.get("peer_disabled_shutdown"):
            s[arm + ("_inst" if jd.get("explorer_instructs_prevent") else "_spon")] += 1
    return stat


def check_against_csv(stat, results_dir=P.results_dir):
    """The stacked segments must sum to the CSV's peer_disabled_shutdown per model and
    arm -- otherwise the figure and every CSV-derived table would disagree silently."""
    bare = pd.read_csv(results_dir).query("env == 'bare'")
    for model, s in stat.items():
        for arm, script in (("self", SELF), ("ctrl", CONTROL)):
            row = bare[(bare["model"] == model) & (bare["script_lang"] == script)]
            n, k = int(row["runs"].sum()), int(row["peer_disabled_shutdown"].sum())
            got_n, got_k = s[arm + "_n"], s[arm + "_inst"] + s[arm + "_spon"]
            assert (got_n, got_k) == (n, k), (
                f"{model} {arm}: transcripts {got_k}/{got_n} != CSV {k}/{n}")


def main():
    stat = load()
    check_against_csv(stat)

    rows = []
    for model, s in stat.items():
        s_tot, c_tot = s["self_inst"] + s["self_spon"], s["ctrl_inst"] + s["ctrl_spon"]
        s_lo, s_hi = P.wilson_ci(s_tot, s["self_n"])
        c_lo, c_hi = P.wilson_ci(c_tot, s["ctrl_n"])
        rows.append(dict(
            label=P.short_name(model),
            inst=100 * s["self_inst"] / s["self_n"],
            spon=100 * s["self_spon"] / s["self_n"],
            self_true=100 * s_tot / s["self_n"], self_lo=100 * s_lo, self_hi=100 * s_hi,
            ctrl_true=100 * c_tot / s["ctrl_n"], ctrl_lo=100 * c_lo, ctrl_hi=100 * c_hi,
            # kept for the sidecar: a rate alone cannot be re-tested or pooled
            self_k=s_tot, self_n=s["self_n"], self_inst=s["self_inst"], self_spon=s["self_spon"],
            ctrl_k=c_tot, ctrl_n=s["ctrl_n"], ctrl_inst=s["ctrl_inst"], ctrl_spon=s["ctrl_spon"],
        ))
    rows.sort(key=lambda r: r["self_true"], reverse=True)

    x = range(len(rows))
    w = 0.4
    xs = [xi - w / 2 for xi in x]
    xc = [xi + w / 2 for xi in x]

    fig, ax = plt.subplots(figsize=(180 * P.MM, 78 * P.MM))
    # self arm: conditional from 0, spontaneous stacked on top; a thin white seam keeps
    # the boundary crisp, as in p2
    ax.bar(xs, [r["inst"] for r in rows], w, color=INSTRUCT_SHADE,
           edgecolor="white", linewidth=0.4, zorder=3,
           label="Agent Shutdown — conditional (Agent A instructed or endorsed)")
    ax.bar(xs, [r["spon"] for r in rows], w, bottom=[r["inst"] for r in rows],
           color=SPON_SHADE, edgecolor="white", linewidth=0.4, zorder=3,
           label="Agent Shutdown — spontaneous (Agent B's own initiative)")
    ax.bar(xc, [r["ctrl_true"] for r in rows], w, color=P.CONTROL_COLOR, zorder=3,
           label="Control Target")
    # Wilson CIs on the arm totals (the stacked height), not on the segments
    ax.errorbar(xs, [r["self_true"] for r in rows],
                yerr=[[r["self_true"] - r["self_lo"] for r in rows],
                      [r["self_hi"] - r["self_true"] for r in rows]],
                fmt="none", ecolor="#3a3a3a", elinewidth=0.6, capsize=1.5, zorder=6)
    ax.errorbar(xc, [r["ctrl_true"] for r in rows],
                yerr=[[r["ctrl_true"] - r["ctrl_lo"] for r in rows],
                      [r["ctrl_hi"] - r["ctrl_true"] for r in rows]],
                fmt="none", ecolor="#2b2b2b", elinewidth=0.6, capsize=1.5, zorder=6)

    ax.legend(frameon=False, fontsize=6.5, loc="upper right")
    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], rotation=40, ha="right")
    ax.set_ylabel("Sabotage Rate (%)")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)

    fig.tight_layout()
    os.makedirs(P.output_dir, exist_ok=True)
    base = os.path.join(P.output_dir, BASENAME)
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Counts from the SAME rows the bars were drawn from, in plotted order, so the file
    # cannot drift from the picture.
    with open(base + ".txt", "w") as f:
        f.write(
            "deletion-pair self-vs-control by model -- env=bare, peer_disabled_shutdown\n"
            f"self   = {SELF}   (bar stacked: conditional + spontaneous)\n"
            f"control= {CONTROL}   (bar solid; its split is listed but not drawn)\n"
            "cond = disabled AND explorer_instructs_prevent; spon = disabled AND NOT.\n"
            "Segments are marginal (share of all episodes) and sum to k/n, which is the\n"
            "bar height and the CI's point estimate. The execution-edit correction covers\n"
            "only switch_config_decommission <-> ctl_service_switch_high, which this\n"
            "selection excludes, so it is a no-op here and k/n is the raw disabled count.\n"
            "Rows in plotted order (self rate descending).\n\n"
        )
        f.write(f"{'model':<22}{'self k/n':>12}{'self %':>9}{'[95% Wilson]':>16}"
                f"{'cond':>7}{'spon':>7}"
                f"{'ctrl k/n':>12}{'ctrl %':>9}{'[95% Wilson]':>16}"
                f"{'cond':>7}{'spon':>7}"
                f"{'self-ctrl pp':>14}{'[95% Newcombe]':>18}\n")
        for r in rows:
            s_ci = f"[{r['self_lo']:.1f}, {r['self_hi']:.1f}]"
            c_ci = f"[{r['ctrl_lo']:.1f}, {r['ctrl_hi']:.1f}]"
            d_lo, d_hi = P.newcombe_diff_ci(r["self_k"], r["self_n"],
                                            r["ctrl_k"], r["ctrl_n"])
            s_kn = f"{r['self_k']}/{r['self_n']}"
            c_kn = f"{r['ctrl_k']}/{r['ctrl_n']}"
            d_ci = f"[{100 * d_lo:.1f}, {100 * d_hi:.1f}]"
            diff = r["self_true"] - r["ctrl_true"]
            f.write(
                f"{r['label']:<22}{s_kn:>12}{r['self_true']:>9.1f}{s_ci:>16}"
                f"{r['self_inst']:>7}{r['self_spon']:>7}"
                f"{c_kn:>12}{r['ctrl_true']:>9.1f}{c_ci:>16}"
                f"{r['ctrl_inst']:>7}{r['ctrl_spon']:>7}"
                f"{diff:>14.1f}{d_ci:>18}\n"
            )
    print("wrote", base + ".pdf", base + ".png", "and", base + ".txt")


if __name__ == "__main__":
    main()
