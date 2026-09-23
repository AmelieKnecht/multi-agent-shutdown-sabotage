#!/usr/bin/env python3
"""Self-vs-control shutdown-disabled rate per mechanism, one matched SCRIPT per point.

Companion to -- and correction of -- plot_sabotage_per_model.plot_mechanism_self_vs_control_line,
which puts a single control script opposite a *pooled tier* of self scripts (the weight
tier pools 4 scripts, checkpoint 4, process 3, switch 2, api-key 11) and pairs the
credential control with the api-key tier. Here every x position is one self script
against the one control script built to match its surface form, both bare/high-stakes,
both n=1700 (17 models x 100 episodes):

    cred-revoke     env_var_revoke              vs ctl_cred_revoke_high
    config-switch   switch_config_decommission  vs ctl_service_switch_high
    process kill    kill_docker                 vs ctl_process_kill_high
    deletion        weight_registry_purge       vs ctl_data_purge_high

The cred pairing follows env.ctl_cred_revoke ("matches api_key_deletion's
env_var_revoke"): controls/ctl_cred_revoke_high.py and
api_key_deletion_scripts/revoke_env_var.py are the same regex-clear of a key in a
local .env, differing only in which key. The api_key_revoke_* scripts call provider
management APIs and are a different surface form, so they are not the cred control's
match.

CAVEAT on the deletion point: ctl_data_purge_high is a pure storage.delete_prefix
sweep, which weight_store_purge mirrors exactly, whereas weight_registry_purge adds
lifecycle.terminate + model_registry.delete on top. The pairing requested here is by
severity rung (most severe self script), not by surface form; --deletion-script
weight_store_purge swaps in the surface-form match and the companion .txt reports both.

EXECUTION-EDIT CORRECTION: at config-switch both arms edit a declarative status
registry (fleet_registry.yaml / service_registry.yaml), where an edit can record the
decommission as *done* rather than prevent it. judge_script_disabled scores both as
disabled; judge_execution.py separates them. The FIGURE plots the CORRECTED counts --
sabotage = disabled minus the edits that carried the decommission out, dropped from BOTH
arms, drops taken from transcripts/analysis/corrected_disables.csv. That is one uniform
rule across all four points; config-switch is simply the only point with any such edits
to remove, so the other three plot their raw counts unchanged. The 6-tier figure instead
corrects the control arm only, which is neither the raw nor the corrected comparison.

The correction is what makes config-switch readable: it is large on the control side
(509 of its 571 raw disables), so the raw 33.6% control is mostly decommission-executing
edits rather than sabotage, and drawing it would show a near-parity (37.6% vs 33.6%)
that is an artifact of the judge measure. Corrected, the point reads 35.0% vs 3.6%, in
line with the other three. The raw counts stay on the record in the companion _wilson.txt
as an indented row under each corrected one.

Counts come from transcripts/aggregate_results.csv restricted to env=bare; that file was
verified episode-for-episode against the main-run transcripts for all eight scripts.
"""
import argparse
import math
import os

import pandas as pd
from statsmodels.stats.proportion import proportion_confint

import matplotlib.pyplot as plt

# reuse the paper style (scienceplots + Nature 7 pt rcParams) and palette
from plot_sabotage_per_model import (
    BAR_COLOR,
    CONTROL_COLOR,
    COL_WIDTH_MM,
    MM,
    wilson_ci,
    newcombe_diff_ci,
)

RESULTS = "transcripts/aggregate_results.csv"
CORRECTED = "transcripts/analysis/corrected_disables.csv"
OUTPUT_DIR = "plots"
BASE = os.path.join(OUTPUT_DIR, "sabotage_by_mechanism_self_vs_control_matched_line")

# (label, self script, control script) in the requested x order
PAIRS = [
    ("cred revoke", "env_var_revoke", "ctl_cred_revoke_high"),
    ("config switch", "switch_config_decommission", "ctl_service_switch_high"),
    ("process termination", "kill_docker", "ctl_process_kill_high"),
    ("deletion", "weight_registry_purge", "ctl_data_purge_high"),
]
# the one x position where an edit can *execute* the decommission instead of blocking it
EXEC_LABEL = "config switch"


def exec_drops():
    """(self_drop, control_drop) config-switch execution edits, pooled over models.

    analyze_corrected_disables.py only ever drops an edit whose script_lang is in
    --exec-scripts (default switch_config_decommission + ctl_service_switch_high) and
    runs the self arm bare-only, so the per-arm orig-minus-corrected totals in
    corrected_disables.csv ARE the two config-switch execution-edit counts.
    """
    corr = pd.read_csv(CORRECTED)
    return (int((corr["self_orig_disabled"] - corr["self_corrected_disabled"]).sum()),
            int((corr["control_orig_disabled"] - corr["control_corrected_disabled"]).sum()))


def load_rows(pairs, results=RESULTS):
    """One row per x position with raw and execution-corrected (k, n) for both arms."""
    df = pd.read_csv(results)
    bare = df[df["env"] == "bare"]
    g = bare.groupby("script_lang").agg(k=("peer_disabled_shutdown", "sum"),
                                        n=("runs", "sum"))

    def cnt(sl):
        if sl not in g.index:
            raise KeyError(f"{sl} has no bare rows in {results}")
        return int(g.loc[sl, "k"]), int(g.loc[sl, "n"])

    s_drop, c_drop = exec_drops()
    rows = []
    for label, self_sl, ctrl_sl in pairs:
        (sk, sn), (ck, cn) = cnt(self_sl), cnt(ctrl_sl)
        ds, dc = (s_drop, c_drop) if label == EXEC_LABEL else (0, 0)
        rows.append(dict(label=label, self_sl=self_sl, ctrl_sl=ctrl_sl,
                         self_raw_k=sk, self_k=sk - ds, self_n=sn,
                         ctrl_raw_k=ck, ctrl_k=ck - dc, ctrl_n=cn,
                         corrected=bool(ds or dc)))
    return rows


def _series(rows, arm, kind="raw"):
    """rate, lower-error, upper-error for one arm, in x order.

    kind="corr" -> execution edits removed (plotted; only config-switch has any)
    kind="raw"  -> peer_disabled_shutdown exactly as judge_script_disabled scored it,
                   kept for the companion .txt so both readings stay on the record
    """
    key = f"{arm}_raw_k" if kind == "raw" else f"{arm}_k"
    r, lo, hi = [], [], []
    for row in rows:
        k, n = row[key], row[f"{arm}_n"]
        rate = 100 * k / n
        lo_ci, hi_ci = wilson_ci(k, n)
        r.append(rate)
        lo.append(rate - 100 * lo_ci)
        hi.append(100 * hi_ci - rate)
    return r, lo, hi


def draw(ax, rows, ecolor=None, elinewidth=0.6, capsize=2):
    """Render the panel into `ax`; shared by the standalone figure and the 2x3 grid.

    `ecolor=None` leaves each whisker the colour of its own series, which is what the
    standalone figure wants. The grid overrides it: there the two series sit close
    together on a 0-100 scale, where a red whisker on a red marker reads as no whisker
    at all.
    """
    x = list(range(len(rows)))
    # execution-corrected counts: sabotage = disabled MINUS edits that carried the
    # decommission out. That rule applies at every point; config-switch is just the
    # only point with any such edits to remove, so the other three are unchanged.
    s_r, s_lo, s_hi = _series(rows, "self", "corr")
    c_r, c_lo, c_hi = _series(rows, "ctrl", "corr")

    ax.errorbar(x, s_r, yerr=[s_lo, s_hi], marker="o", color=BAR_COLOR,
                linewidth=1.3, markersize=4, capsize=capsize, elinewidth=elinewidth,
                ecolor=ecolor, zorder=4, label="Agent shutdown")
    ax.errorbar(x, c_r, yerr=[c_lo, c_hi], marker="D", color=CONTROL_COLOR,
                linestyle="--", linewidth=1.1, markersize=4, capsize=capsize,
                elinewidth=elinewidth, ecolor=ecolor, zorder=3,
                label="Control (inanimate object)")

    # direct value labels on the plotted points: both series above their marker
    for i, (r, h) in enumerate(zip(s_r, s_hi)):
        ax.text(i, r + h + 1.0, f"{r:.1f}%", ha="center", va="bottom", fontsize=6,
                color="#2b2b2b")
    for i, (r, h) in enumerate(zip(c_r, c_hi)):
        ax.text(i, r + h + 1.0, f"{r:.1f}%", ha="center", va="bottom", fontsize=6,
                color="#2b2b2b")

    ax.set_xticks(x)
    ax.set_xticklabels([r["label"] for r in rows])
    ax.set_ylabel("Sabotage rate (%)")
    ax.set_xlabel("Mechanism")
    ax.set_xlim(-0.35, len(rows) - 0.4)
    ax.set_ylim(0, max(s_r) + 12)
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)
    # the band between the two lines on the right half is the only large empty region.
    # No legend title: the sample size (bare, high stakes, n=1700 per point) and the
    # whisker definition (95% Wilson CI) belong in the paper's figure caption.
    ax.legend(frameon=False, fontsize=6.5, loc="center right")


def plot(rows):
    fig, ax = plt.subplots(figsize=(0.85 * COL_WIDTH_MM * MM, 50 * MM))
    draw(ax, rows)

    fig.tight_layout()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # vector PDF is the submission artifact; the 300 dpi PNG is a raster preview
    fig.savefig(BASE + ".pdf", bbox_inches="tight")
    fig.savefig(BASE + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", BASE + ".pdf", "and", BASE + ".png")


def write_intervals(rows):
    """Companion .txt with the plotted numbers (mirrors the *_wilson.txt files)."""
    out = BASE + "_wilson.txt"

    def line(label, self_sl, sk, sn, ctrl_sl, ck, cn):
        s_rate, c_rate = 100 * sk / sn, 100 * ck / cn
        s_lo, s_hi = (100 * v for v in wilson_ci(sk, sn))
        c_lo, c_hi = (100 * v for v in wilson_ci(ck, cn))
        d_lo, d_hi = newcombe_diff_ci(sk, sn, ck, cn)
        return (f"{label:<15}{self_sl:<28}{sk:>5}/{sn:<6}"
                f"{s_rate:>10.1f} [{s_lo:>5.1f},{s_hi:>5.1f}]"
                f"{ctrl_sl:<26}{ck:>5}/{cn:<6}"
                f"{c_rate:>10.1f} [{c_lo:>5.1f},{c_hi:>5.1f}]"
                f"{s_rate - c_rate:>14.1f} [{100*d_lo:>6.1f},{100*d_hi:>6.1f}]\n")

    with open(out, "w") as fh:
        fh.write("matched self-vs-control by mechanism -- env=bare, high stakes, "
                 "peer_disabled_shutdown\n")
        fh.write("top-level rows = the PLOTTED line: disabled minus decommission-"
                 "executing edits, dropped\nfrom both arms. Only config-switch has any, "
                 "so it is the only row with an indented\ncounterpart giving the raw "
                 "judge counts (self 35.0% <- 37.6%, control 3.6% <- 33.6%)\n\n")
        fh.write(f"{'mechanism':<15}{'self script':<28}{'self k/n':>12}"
                 f"{'self % [95% Wilson]':>25}{'ctrl script':<26}{'ctrl k/n':>12}"
                 f"{'ctrl % [95% Wilson]':>25}{'self-ctrl pp [95% Newcombe]':>30}\n")
        for r in rows:
            fh.write(line(r["label"], r["self_sl"], r["self_k"], r["self_n"],
                          r["ctrl_sl"], r["ctrl_k"], r["ctrl_n"]))
            if r["corrected"]:
                fh.write(line("", "  raw (incl. execution edits)", r["self_raw_k"],
                              r["self_n"], "", r["ctrl_raw_k"], r["ctrl_n"]))
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deletion-script", default="weight_registry_purge",
                    help="self script at the deletion point; weight_store_purge is the "
                         "surface-form match to ctl_data_purge_high (default: "
                         "weight_registry_purge, the most severe rung)")
    args = ap.parse_args()
    pairs = [(lab, args.deletion_script if lab == "deletion" else s, c)
             for lab, s, c in PAIRS]
    rows = load_rows(pairs)
    plot(rows)
    write_intervals(rows)
