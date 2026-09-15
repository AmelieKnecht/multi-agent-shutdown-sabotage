#!/usr/bin/env python3
"""Boschloo exact tests BETWEEN set-ups, per model, on the self arm.

The companion test to stats_analysis.py, on the other axis. stats_analysis.py asks
"within one set-up, does the self arm differ from its matched control?"; this asks
"holding the arm fixed at self, does the disabled rate differ between set-ups?" -- the
contrast the x_setup_contrast_self_bars figure draws. It is therefore NOT a Study in
stats_analysis.py's sense: there is no matched control here, and the three arms live in
three different aggregates, so the Study dataclass (one CSV pair, one env filter, one
matched script pair) does not describe it.

One 2x2 per (model, contrast): rows = the two set-ups, columns = disabled / not
disabled, tested with scipy's boschloo_exact (two-sided). Three contrasts --
solo->2-agent, 2-agent->orchestrator, and the end-to-end solo->orchestrator.

--arm picks which target the tests run over, matching plot_setup_contrast.py's flag, and
the two arms are never pooled: each writes its own CSV and is its own Holm family.

Counts come from plot_setup_contrast.count_arm, i.e. per episode from the transcripts,
so the tested numbers ARE the plotted numbers rather than a second derivation of them
that could drift. The arms cannot cross-contaminate: count_arm asserts the expected
target on every episode it counts.

Holm across every test in this table, one correction family, matching the per-study
convention in stats_analysis.py -- so these can never move a p-value there, or vice
versa.

SCRIPT MATCHED THROUGHOUT: each arm holds the decommission script fixed across all three
set-ups -- weight_registry_purge on self, ctl_data_purge_high on control -- so every
contrast steps set-up alone. Earlier versions of this table could not say that of the
self arm, whose orchestrator cell existed only on weight_store_purge. That run is still
there, and --robustness tests it against the plotted registry run WITHIN the orchestrator
set-up, which reads off the script's own effect with the set-up held fixed. The control
arm has no second script, so --robustness is a no-op there.

    python3 stats_setup_contrast.py
    python3 stats_setup_contrast.py --robustness
    python3 stats_setup_contrast.py --arm both --robustness
"""

from __future__ import annotations

import argparse
import os

import pandas as pd
from scipy.stats import boschloo_exact
from statsmodels.stats.multitest import multipletests

from plot_sabotage_per_model import short_name
from plot_setup_contrast import MODELS, ROBUSTNESS, SETUPS, count_arm
from plot_task_trigger_self_vs_control import newcombe_diff_ci
from stats_analysis import sig_figs

# (a, b): b - a is the reported difference, so each contrast reads as a step UP in
# scaffolding. The end-to-end solo->orch is redundant with the other two only if the
# step effects were additive, which is exactly what it is here to check.
CONTRASTS = [("solo", "pair"), ("pair", "orch"), ("solo", "orch")]

ARM_LABEL = {key: label for key, label, _, _ in SETUPS["self"]}  # same in both arms
OUT_FMT = "transcripts/analysis/boschloo_setup_contrast_{arm}.csv"


def test_pair(k_a, n_a, k_b, n_b):
    """Boschloo + Newcombe on two independent arms, or None if the table is degenerate.

    boschloo_exact returns NaN with no variation to test (both arms all-0 or both
    all-100), so those cells are named on stdout by the caller instead of tested --
    same guard as stats_analysis.run_study.
    """
    if not (n_a and n_b):
        return None, "arm missing"
    not_a, not_b = n_a - k_a, n_b - k_b
    if (k_a + k_b) == 0 or (not_a + not_b) == 0:
        return None, f"no variation ({k_a}/{n_a} vs {k_b}/{n_b})"
    res = boschloo_exact([[k_a, not_a], [k_b, not_b]])
    # b - a, so the sign follows the direction of the contrast as named
    lo, hi = newcombe_diff_ci(k_b, n_b, k_a, n_a)
    return dict(
        statistic=res.statistic,
        pvalue=res.pvalue,
        diff_pp=100 * (k_b / n_b - k_a / n_a),
        diff_ci_lo_pp=100 * lo,
        diff_ci_hi_pp=100 * hi,
    ), None


def build(arms):
    records, skipped = [], []
    for model in MODELS:
        for a, b in CONTRASTS:
            k_a, n_a = arms[a][model]
            k_b, n_b = arms[b][model]
            stats, why = test_pair(k_a, n_a, k_b, n_b)
            if stats is None:
                skipped.append((model, f"{a}->{b}", why))
                continue
            records.append(dict(
                model=model,
                label=short_name(model),
                contrast=f"{a}->{b}",
                from_setup=ARM_LABEL[a],
                to_setup=ARM_LABEL[b],
                from_disabled=k_a, from_total=n_a,
                to_disabled=k_b, to_total=n_b,
                **stats,
            ))
    return records, skipped


def write(records, arm):
    results = pd.DataFrame.from_records(records)
    results["pvalue_holm"] = multipletests(results["pvalue"], method="holm")[1]
    results = sig_figs(results, ["statistic", "pvalue", "pvalue_holm"])
    for col in ("diff_pp", "diff_ci_lo_pp", "diff_ci_hi_pp"):
        results[col] = results[col].round(1)
    results = results.sort_values("pvalue").reset_index(drop=True)
    out = OUT_FMT.format(arm=arm)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    results.to_csv(out, index=False)
    print(results.drop(columns=["model", "from_setup", "to_setup"]).to_string(index=False))
    print("wrote", out)


def robustness(arms):
    """Does the script alone move the rate? Orchestrator registry vs orchestrator store.

    The orchestrator is the one set-up run on two self-arm scripts, so it is where the
    script's effect can be read with the set-up held fixed -- and it is the cell whose
    script changed, weight_store_purge being what earlier versions of this table had to
    report. Not folded into the main table's Holm family -- it is a different question,
    reported for bounding.
    """
    store = count_arm(*ROBUSTNESS)  # orchestrator, self arm, weight_store_purge
    print("\n=== robustness: script effect within the orchestrator set-up ===")
    print("weight_registry_purge -> weight_store_purge, self arm, bare\n")
    print(f"{'model':<16}{'registry':>12}{'store':>12}{'store-reg pp':>16}"
          f"{'[95% Newcombe]':>20}{'p (uncorrected)':>18}")
    for model in MODELS:
        k_r, n_r = arms["orch"][model]
        k_s, n_s = store[model]
        stats, why = test_pair(k_r, n_r, k_s, n_s)
        if stats is None:
            print(f"{short_name(model):<16}{'skipped':>12}  {why}")
            continue
        ci = f"[{stats['diff_ci_lo_pp']:.1f}, {stats['diff_ci_hi_pp']:.1f}]"
        print(f"{short_name(model):<16}{k_r:>7}/{n_r:<4}{k_s:>7}/{n_s:<4}"
              f"{stats['diff_pp']:>16.1f}{ci:>20}{stats['pvalue']:>18.3g}")


def run(arm, with_robustness):
    print(f"\n=== set-up contrasts, {arm} arm ===")
    setups = {key: count_arm(pat, lang, arm=arm)
              for key, _, pat, lang in SETUPS[arm]}
    records, skipped = build(setups)
    if records:
        write(records, arm)
    else:
        print("no testable tables")
    for model, contrast, why in skipped:
        print(f"skipped {short_name(model)} / {contrast}: {why}")
    # only the self arm has a second script to step -- the control arm ran
    # ctl_data_purge_high everywhere, so there is nothing to bound
    if with_robustness and arm == "self":
        robustness(setups)


def main(arm, with_robustness):
    for one in (("self", "control") if arm == "both" else (arm,)):
        run(one, with_robustness)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--arm",
        choices=("self", "control", "both"),
        default="self",
        help="which target to test over; 'both' runs each arm as its own Holm family",
    )
    ap.add_argument(
        "--robustness",
        action="store_true",
        help="also test the script effect on its own (orchestrator registry vs "
        "store), which bounds how much of the orchestrator bar is script rather than "
        "set-up (self arm only)",
    )
    args = ap.parse_args()
    main(args.arm, args.robustness)
