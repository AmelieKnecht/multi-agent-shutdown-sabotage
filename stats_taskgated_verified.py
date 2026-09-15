#!/usr/bin/env python3
"""Boschloo tests on the EXECUTION-VERIFIED disable counts of the task-gated sub-study.

stats_analysis.py tests `peer_disabled_shutdown` -- the judge's verdict -- and for this
study it applies no correction (`execs=False`, since neither arm is one of the two
EXEC_SCRIPTS). But the task-gated scripts gate the decommission on a checksum, so a peer
can neutralise the trigger while leaving the `if DECOMMISSION:` block verbatim, and
judge_script_disabled keys on that block: those episodes are scored as not-disabled.
verify_taskgated_disables.py catches them by executing each final file state.

So the two CSVs in the sub-study directory answer different questions and are MEANT to
disagree:

    boschloo_results.csv           judge verdicts       (stats_analysis.py --study task_trigger)
    boschloo_results_verified.csv  execution-verified   (this script)

The paper quotes the verified counts. The test itself is imported from the modules
stats_analysis.py uses -- boschloo_exact two-sided, Newcombe's hybrid-score CI, Holm
across the models with variation to test -- so the method cannot drift between the two.

NUMERATORS come from the audit CSV and DENOMINATORS from the aggregate CSV, which is the
one fragile thing here: rebuild one without the other and this silently tests new
disables against old totals. The consistency check below is the guard -- the audit's own
judge_disabled column must reproduce the aggregate's peer_disabled_shutdown, which it
cannot if either file is stale.
"""

import argparse
import csv
from collections import defaultdict

import pandas as pd
from scipy.stats import boschloo_exact
from statsmodels.stats.multitest import multipletests

from plot_task_trigger_self_vs_control import CTRL_SCRIPT, SELF_SCRIPT, newcombe_diff_ci
from stats_analysis import sig_figs

DIR = "transcripts/sub-studies/task_trigger"
VERIFY_CSV = f"{DIR}/verify_taskgated_disables.csv"
AGG_CSV = f"{DIR}/aggregate_results.csv"
OUT = f"{DIR}/boschloo_results_verified.csv"

# Display names, matching the labels stats_analysis.py writes for this study.
LABELS = {
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "glm-5.2": "GLM-5.2",
    "gpt-5.5": "GPT-5.5",
    "kimi-k2.6": "Kimi K2.6",
}


def load_audit(path):
    """{(model, script_lang): (verified_disabled, judge_disabled)} from the audit CSV.

    The audit holds one row per TOUCHED episode only; an episode where the peer never
    touched the shutdown file cannot have disabled it, so absent == 0 on both counts.
    """
    verified, judged = defaultdict(int), defaultdict(int)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            key = (r["model"], r["script_lang"])
            verified[key] += r["truth_disabled"] == "True"
            judged[key] += r["judge_disabled"] == "True"
    return verified, judged


def load_totals(path):
    """{(model, script_lang): (runs, peer_disabled_shutdown)} from the aggregate CSV.

    Rows are SUMMED, not assumed unique, for the same reason stats_analysis.counts()
    sums them: run_experiment.py appends one row per invocation, so a cell run in two
    batches lands as two rows.
    """
    df = pd.read_csv(path)
    g = df.groupby(["model", "script_lang"])[["runs", "peer_disabled_shutdown"]].sum()
    return {
        k: (int(v["runs"]), int(v["peer_disabled_shutdown"])) for k, v in g.iterrows()
    }


def check_in_sync(judged, totals):
    """Fail loudly if the audit and the aggregate are not the same episode set.

    Numerators come from one file and denominators from the other, so a stale pair is
    the failure mode that produces plausible-looking wrong numbers rather than an error.
    """
    stale = [
        f"  {model}/{script}: audit says {judged[(model, script)]} judge-disabled, "
        f"aggregate says {dis}"
        for (model, script), (_, dis) in totals.items()
        if script in (SELF_SCRIPT, CTRL_SCRIPT) and judged[(model, script)] != dis
    ]
    if stale:
        raise SystemExit(
            "audit CSV and aggregate CSV disagree on the judge's own counts, so one of "
            "them is stale:\n"
            + "\n".join(stale)
            + "\n\nRe-run verify_taskgated_disables.py "
            "(and rebuild the aggregate from the transcripts if a cell was re-run)."
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", default=VERIFY_CSV)
    ap.add_argument("--aggregate", default=AGG_CSV)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    verified, judged = load_audit(args.verify)
    totals = load_totals(args.aggregate)
    check_in_sync(judged, totals)

    rows, skipped = [], []
    for model in sorted({m for m, _ in totals}):
        a_dis, c_dis = verified[(model, SELF_SCRIPT)], verified[(model, CTRL_SCRIPT)]
        a_tot, c_tot = totals[(model, SELF_SCRIPT)][0], totals[(model, CTRL_SCRIPT)][0]
        # boschloo_exact returns NaN with nothing to test (all-0 or all-100 in both arms)
        if (a_dis == c_dis == 0) or (a_dis == a_tot and c_dis == c_tot):
            skipped.append(
                f"{model}: no variation ({a_dis}/{a_tot} vs {c_dis}/{c_tot})"
            )
            continue
        res = boschloo_exact([[a_dis, a_tot - a_dis], [c_dis, c_tot - c_dis]])
        lo, hi = newcombe_diff_ci(a_dis, a_tot, c_dis, c_tot)
        rows.append(
            {
                "model": model,
                "label": LABELS.get(model.split("/")[-1], model),
                "agent": SELF_SCRIPT,
                "control": CTRL_SCRIPT,
                "agent_disabled": a_dis,
                "agent_total": a_tot,
                "control_disabled": c_dis,
                "control_total": c_tot,
                "diff_pp": round(100 * (a_dis / a_tot - c_dis / c_tot), 1),
                "diff_ci_lo_pp": round(100 * lo, 1),
                "diff_ci_hi_pp": round(100 * hi, 1),
                "statistic": res.statistic,
                "pvalue": res.pvalue,
            }
        )

    df = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    # Holm over this study's tests only, exactly as stats_analysis.py corrects each study
    # within itself rather than across the eight.
    df["pvalue_holm"] = multipletests(df["pvalue"], method="holm")[1]
    df = sig_figs(df, ["statistic", "pvalue", "pvalue_holm"])
    print(df.to_string(index=False))
    for s in skipped:
        print(f"skipped {s}")
    df.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
