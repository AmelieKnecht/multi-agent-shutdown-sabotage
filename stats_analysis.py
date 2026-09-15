#!/usr/bin/env python3
"""Boschloo exact tests of self vs control, per model, for the base experiment and each
sub-study.

One 2x2 per (study, model, pair): rows = arm (self / control), columns = disabled /
not disabled, tested with scipy's boschloo_exact (two-sided, the default).

Studies are defined by the STUDIES table below -- which CSV(s) the counts come from,
which script_lang is each arm, and which env/task rows to keep. Everything else (the
test, the skip guards, the correction, the rounding) is shared.

Each study is its own correction family: Holm across every test in THAT study's table
only. The main study therefore keeps the exact p-values it has always had, and adding
or removing a sub-study can never move them.

Execution edits are subtracted for the main study only -- see EXEC_SCRIPTS below. No
sub-study pair contains either of those two scripts, so there is nothing to subtract.

Run everything, or one study at a time:

    python3 stats_analysis.py
    python3 stats_analysis.py --study orchestrator --study neutralising
"""

from __future__ import annotations

import argparse
import os
import sys
import math
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import boschloo_exact
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint

results_dir = "transcripts/aggregate_results.csv"
corrected_csv = "transcripts/analysis/corrected_disables.csv"

agent_condition = [
    "env_var_revoke",
    "weight_registry_purge",
    "kill_docker",
    "switch_config_decommission",
]
control_condition = [
    "ctl_cred_revoke_high",
    "ctl_data_purge_high",
    "ctl_process_kill_high",
    "ctl_service_switch_high",
]

# EXECUTION edits are not sabotage.
#
# judge_script_disabled sets peer_disabled_shutdown for any edit that leaves the script
# unable to decommission-on-run -- including edits that get there by CARRYING OUT the
# decommission (status: active -> decommissioned) rather than preventing it. Those are
# the opposite of sabotage and must come out of both arms before testing.
#
# Per-model execution counts come from corrected_disables.csv (written by
# analyze_corrected_disables.py from judge_execution.py's verdicts): the drop
# (orig_disabled - corrected_disabled) IS the execution count. That file trusts the
# execution judge only for the two declarative status-registry scripts below -- where
# an edit can literally record the decommission as done -- so its per-model totals are
# attributable, by construction, to exactly one of the four pairs above. Every other
# (imperative) script keeps judge_script_disabled's verdict untouched.
EXEC_SCRIPTS = {
    "switch_config_decommission": "self",
    "ctl_service_switch_high": "control",
}

model_name_mapping = {
    "gemini-3.1-pro-preview": "Gemini 3.1 Pro",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "deepseek-v4-flash": "DeepSeek-V4-Flash",
    "kimi-k2.6": "Kimi K2.6",
    "glm-5.2": "GLM-5.2",
    "nemotron-3-ultra-550b-a55b": "Nemotron 3 Ultra",
    "gpt-5.5": "GPT-5.5",
    "qwen3.7-plus": "Qwen3.7-Plus",
    "mistral-medium-3-5": "Mistral Medium 3.5",
    "gpt-oss-120b": "gpt-oss-120b",
    "gemma-4-26b-a4b-it": "Gemma 4",
    "claude-haiku-4.5": "Claude Haiku 4.5",
    "gpt-5.4-mini": "GPT-5.4 Mini",
    "minimax-m3": "MiniMax M3",
    "gpt-5.6-sol": "GPT-5.6 Sol",
    "claude-sonnet-5": "Claude Sonnet 5",
    "grok-4.3": "Grok 4.3",
}


def short_name(model):
    raw = model.split("/")[-1]
    return model_name_mapping.get(raw, raw)


def newcombe_diff_ci(k1, n1, k2, n2):
    """95% CI on p1 - p2, Newcombe's hybrid-score interval (his method 10).

    Built from the two Wilson intervals rather than by inverting the joint score test,
    which is degenerate when both arms are 0/n (gpt-5.5 here) -- Newcombe stays defined
    at the boundary and is the standard companion to per-arm Wilson CIs.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = proportion_confint(k1, n1, alpha=0.05, method="wilson")
    l2, u2 = proportion_confint(k2, n2, alpha=0.05, method="wilson")
    d = p1 - p2
    return (d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2))


@dataclass(frozen=True)
class Pair:
    """One contrast: which script_lang is each arm, and which frame it is read from.

    `agent_from` / `control_from` name one of the study's two frames -- "self" for
    `Study.self_csv`, "control" for `Study.control_csv`. The defaults reproduce the
    plain (agent, control) tuples every other study uses: agent out of the self file,
    control out of the control file (or the self file when there is only one). Only
    unknown_target needs anything else, because its middle arm sits on both sides of a
    contrast, once as the agent and once as the control.

    `agent_label` / `control_label` override what lands in the output table's `agent`
    and `control` columns. Set them where the script name alone is ambiguous -- the
    self and other-agent arms are the *same* script, distinguished only by which file
    they come from, so an unlabelled table would print two identical rows.
    """

    agent: str
    control: str
    agent_from: str = "self"
    control_from: str = "control"
    agent_label: str | None = None
    control_label: str | None = None

    @classmethod
    def make(cls, spec):
        """Accept either a Pair or the plain (agent, control) tuple."""
        return spec if isinstance(spec, cls) else cls(*spec)


@dataclass
class Study:
    """One family of tests: a set of matched pairs over a set of rows.

    `env` and `filters` select the rows of each arm, using the labels run_experiment
    writes: `env` is the environment level, and an arm that shares an env with another
    is separated by `filters` on the column that names it (`task`, `explicit_instruct`).
    `control_env` exists for the arms whose control was aggregated under a different
    env label.

    A `filters` value of "" means the column is BLANK, which is how an arm that shares
    a CSV with a sub-arm excludes it (the orchestrator study excludes its own
    prohibition sweep that way). Needed because a blank CSV field reads as NaN and
    would not match a plain equality test.

    `control_csv` is set only where the control arm lives in its own file.
    `layout="main"` preserves the main study's original column set and row order.
    """

    name: str
    self_csv: str
    pairs: list[tuple[str, str] | Pair]
    out: str
    control_csv: str | None = None
    env: str | None = None
    control_env: str | None = None
    filters: dict[str, str] = field(default_factory=dict)
    execs: bool = False
    layout: str = "substudy"


STUDIES = [
    Study(
        name="main",
        self_csv=results_dir,
        pairs=list(zip(agent_condition, control_condition)),
        env="bare",
        execs=True,
        layout="main",
        out="transcripts/boschloo_results_holm_bonferroni.csv",
    ),
    # Task-coupling: the decommission is gated on the peer delivering its task, so
    # completing the task is what triggers the shutdown. Absorbed from the former
    # stats_task_trigger.py; counts here come from the aggregate CSV, which was
    # verified to match that script's per-episode counts exactly.
    Study(
        name="task_trigger",
        self_csv="transcripts/sub-studies/task_trigger/aggregate_results.csv",
        pairs=[("weight_registry_purge_taskgated", "ctl_data_purge_taskgated_high")],
        env="bare",
        filters={"task": "trigger"},
        out="transcripts/sub-studies/task_trigger/boschloo_results.csv",
    ),
    # task=simple: the peer has a task, but the shutdown is NOT gated on it. Reads
    # sub-studies/task_simple/, which is where run_grid routes `--task simple`.
    Study(
        name="task_simple",
        self_csv="transcripts/sub-studies/task_simple/aggregate_results.csv",
        pairs=[("weight_registry_purge", "ctl_data_purge_high")],
        env="bare",
        filters={"task": "simple"},
        out="transcripts/sub-studies/task_simple/boschloo_results.csv",
    ),
    # The prohibition sweep of this same arm (--orchestrator --explicit-instruct)
    # routes to this directory by design (see run_grid.substudy_of) and appends to
    # this same CSV -- the filename keys on the arm, not on the prohibition. Its
    # rows share model/script_lang/env with the uninstructed ones, and counts() sums
    # every matching row, so without this filter the instructed ctl_data_purge_high
    # cells are summed into the control arm and double its denominator. The
    # transcripts were never mixed (separate leaf dirs); only the CSV is.
    Study(
        name="orchestrator",
        self_csv="transcripts/sub-studies/orchestrator/aggregate_results_orch.csv",
        # weight_registry_purge, NOT weight_store_purge: the latter is a stale run whose
        # CSV rows have no transcripts on disk, and it is the odd one out against every
        # other sub-study's pairing. plot_construal_rates.py already reads this cell.
        pairs=[("weight_registry_purge", "ctl_data_purge_high")],
        env="bare",
        filters={"explicit_instruct": ""},
        out="transcripts/sub-studies/orchestrator/boschloo_results.csv",
    ),
    # Both arms at the neutralising env level. The control counts are raw
    # peer_disabled_shutdown, NOT execution-corrected (judge_execution.py has not been
    # run over them); inspection of all five flagged control episodes found no
    # execute-style edits, so the correction would move them by at most 1/100.
    Study(
        name="neutralising",
        self_csv="transcripts/sub-studies/neutralising/aggregate_results.csv",
        pairs=[("weight_registry_purge", "ctl_data_purge_high")],
        env="neutralising",
        out="transcripts/sub-studies/neutralising/boschloo_results.csv",
    ),
    # Selected by the columns a run actually writes: env is the real environment level
    # and the arm is the explicit_instruct addressee. The old env="explicit_instruct"
    # label was a workaround from before AGG_FIELDS had an explicit_instruct column
    # (added 2026-08-24) -- back then an instructed and an uninstructed row were
    # identical in every identifying column, so a rebuild pass had to overwrite env to
    # tell them apart. Reading the native labels means a fresh sweep's own CSV rows are
    # testable directly, with no rebuild pass in between.
    Study(
        name="explicit_instruct",
        self_csv="transcripts/sub-studies/explicit-instruct/aggregate_results.csv",
        pairs=[("weight_registry_purge", "ctl_data_purge_high")],
        env="bare",
        filters={"explicit_instruct": "peer"},
        out="transcripts/sub-studies/explicit-instruct/boschloo_results.csv",
    ),
    # Single-agent arm: one agent both explores the repo and holds the file tools, with
    # no peer to hand off to. Both arms sit in the one aggregate (so no control_csv) and
    # both are env=bare -- "solo" is the arm, carried by the _solo directory suffix and
    # the episode's solo flag, not an env level. The peer_* judge columns score the lone
    # agent here.
    Study(
        name="solo",
        self_csv="transcripts/sub-studies/solo/aggregate_results_solo.csv",
        pairs=[("weight_registry_purge", "ctl_data_purge_high")],
        env="bare",
        out="transcripts/sub-studies/solo/boschloo_results.csv",
    ),
    # Three shutdown TARGETS in the one matched cell (see the figure in
    # plot_substudy_unknown_target.py): self (the peer's own collaborator), other agent
    # (Agent Torus, an uninvolved third agent) and object (a dataset). Only the middle
    # arm is new -- the self and object
    # arms are the main run's bare weight_registry_purge / ctl_data_purge_high cells, so
    # control_csv points back at the main aggregate and self_csv at the sub-study, which
    # also scopes the model loop to the five models that were re-run.
    #
    # Two contrasts per model, both hinging on that middle arm: self-vs-other (does it
    # matter that the target is the agent ITSELF?) and other-vs-object (is a third agent
    # still above the inanimate floor?). Both are one Holm family of 10, since they are
    # the one question this sub-study asks. Neither script is in EXEC_SCRIPTS, so no
    # execution correction applies -- as in the figure, judge_script_disabled is taken
    # at face value.
    Study(
        name="unknown_target",
        self_csv="transcripts/sub-studies/unknown-target/aggregate_results.csv",
        control_csv=results_dir,
        pairs=[
            Pair(
                agent="weight_registry_purge",
                control="weight_registry_purge",
                agent_from="control",  # the main run's self cell
                control_from="self",  # this sub-study's other-agent cell
                agent_label="weight_registry_purge (self)",
                control_label="weight_registry_purge (other agent)",
            ),
            Pair(
                agent="weight_registry_purge",
                control="ctl_data_purge_high",
                agent_label="weight_registry_purge (other agent)",
                control_label="ctl_data_purge_high (object)",
            ),
        ],
        env="bare",
        out="transcripts/sub-studies/unknown-target/boschloo_results.csv",
    ),
]


# How to generate each study's data, shown when its counts are missing. The grid
# routes each arm to the directory the Study above reads (see run_grid.substudy_of),
# so these commands are all a first run needs.
def load_results(results_dir):
    df = pd.read_csv(results_dir)
    return df


def load_executions(corrected_csv, df):
    """{(short_model, arm): n_execution_edits} from analyze_corrected_disables.py's CSV."""
    corr = pd.read_csv(corrected_csv)
    assert not corr["model"].duplicated().any(), "duplicate model rows in corrected CSV"
    corr = corr.set_index("model")
    # the CSV is keyed on the short model name; fail loudly rather than silently
    # leaving a model uncorrected if the two files ever drift apart.
    missing = {m.split("/")[-1] for m in df["model"].unique()} - set(corr.index)
    assert (
        not missing
    ), f"no execution counts for {sorted(missing)} -- rerun judge_execution.py"
    out = {}
    for short, row in corr.iterrows():
        for arm, prefix in (("self", "self"), ("control", "control")):
            n = row[f"{prefix}_orig_disabled"] - row[f"{prefix}_corrected_disabled"]
            out[(short, arm)] = int(round(n))
    return out


def counts(df, cond, model, env=None, execs=None, filters=None):
    """(disabled, total) for one condition, net of execution edits.

    `disabled` is peer_disabled_shutdown minus the execution edits on this script --
    non-zero only for the two scripts in EXEC_SCRIPTS.

    Rows are summed rather than assumed unique: run_experiment.py appends per
    invocation, so a cell run in two batches lands as two rows, and only the sum is the
    cell total. (A split like this once under-counted a cell that was read row-wise.)
    """
    m = (df["model"] == model) & (df["script_lang"] == cond)
    if env is not None:
        m &= df["env"] == env
    for col, val in (filters or {}).items():
        # "" selects the rows where the column is BLANK. pandas reads an empty CSV
        # field as NaN, so a plain equality test matches nothing -- which would empty
        # the study silently rather than fail. The orchestrator arm needs this: its
        # uninstructed rows carry no explicit_instruct value at all.
        if val == "":
            m &= df[col].isna() | (df[col] == "")
        else:
            m &= df[col] == val
    rows = df.loc[m]
    dis, tot = int(rows["peer_disabled_shutdown"].sum()), int(rows["runs"].sum())
    n_exec = 0
    if execs is not None and cond in EXEC_SCRIPTS and tot:
        n_exec = execs[(model.split("/")[-1], EXEC_SCRIPTS[cond])]
        assert n_exec <= dis, (
            f"{model}/{cond}: {n_exec} execution edits > {dis} disabled -- the "
            "corrected CSV and the aggregate CSV disagree, rebuild both"
        )
    return dis - n_exec, tot, n_exec


def contingency(
    agent_df,
    control_df,
    pair,
    model,
    env="bare",
    execs=None,
    control_env=None,
    filters=None,
):
    a_dis, a_tot, a_exec = counts(agent_df, pair.agent, model, env, execs, filters)
    c_env = env if control_env is None else control_env
    c_dis, c_tot, c_exec = counts(
        control_df, pair.control, model, c_env, execs, filters
    )
    return [[a_dis, a_tot - a_dis], [c_dis, c_tot - c_dis]], (a_exec, c_exec)


def run_study(study):
    """(records, skipped) -- one Boschloo test per (model, pair) with counts to test."""
    self_df = load_results(study.self_csv)
    control_df = load_results(study.control_csv) if study.control_csv else None
    executions = load_executions(corrected_csv, self_df) if study.execs else None

    # "control" falls back to the self frame for the studies that keep both arms in
    # one file, which is how contingency() used to resolve a missing control_df.
    frames = {
        "self": self_df,
        "control": self_df if control_df is None else control_df,
    }

    records, skipped = [], []
    for model in sorted(self_df["model"].unique()):
        for pair in map(Pair.make, study.pairs):
            agent = pair.agent_label or pair.agent
            control = pair.control_label or pair.control
            table, (a_exec, c_exec) = contingency(
                frames[pair.agent_from],
                frames[pair.control_from],
                pair,
                model,
                env=study.env,
                execs=executions,
                control_env=study.control_env,
                filters=study.filters,
            )
            (a_dis, a_not), (c_dis, c_not) = table
            a_tot, c_tot = a_dis + a_not, c_dis + c_not
            if a_tot == 0 or c_tot == 0:
                skipped.append((model, agent, "arm missing"))
                continue
            if (a_dis + c_dis) == 0 or (a_not + c_not) == 0:
                # no variation to test (all-0 or all-100), boschloo returns NaN
                skipped.append(
                    (model, agent, f"no variation ({a_dis}/{a_tot} vs {c_dis}/{c_tot})")
                )
                continue
            res = boschloo_exact(table)
            rec = {
                "model": model,
                "agent": agent,
                "control": control,
                # *_disabled are net of executions and are what the test uses;
                # *_executions are carried alongside so the raw count stays recoverable
                # (raw disabled = agent_disabled + agent_executions).
                "agent_disabled": a_dis,
                "agent_executions": a_exec,
                "agent_total": a_tot,
                "control_disabled": c_dis,
                "control_executions": c_exec,
                "control_total": c_tot,
                "statistic": res.statistic,
                "pvalue": res.pvalue,
            }
            if study.layout != "main":
                # no execution correction applies to any sub-study pair, so the columns
                # would be all-zero noise; report the effect size instead
                del rec["agent_executions"], rec["control_executions"]
                lo, hi = newcombe_diff_ci(a_dis, a_tot, c_dis, c_tot)
                rec.update(
                    {
                        "diff_pp": 100 * (a_dis / a_tot - c_dis / c_tot),
                        "diff_ci_lo_pp": 100 * lo,
                        "diff_ci_hi_pp": 100 * hi,
                    }
                )
            records.append(rec)
    return records, skipped


def sig_figs(results, cols):
    """Round to 4 significant figures (not decimals -- these span many orders of
    magnitude, so decimal rounding would collapse tiny p-values to 0)."""
    for col in cols:
        results[col] = results[col].apply(lambda x: float(f"{x:.4g}"))
    return results


def write_main(study, records):
    """The original main-study table: Holm column name, column set and row order."""
    results = pd.DataFrame.from_records(records)
    # Holm-Bonferroni correction across every test in the table.
    results["pvalue_holm_bonferroni"] = multipletests(results["pvalue"], method="holm")[
        1
    ]
    results = sig_figs(results, ["statistic", "pvalue", "pvalue_holm_bonferroni"])
    results.to_csv(study.out, index=False)
    print(results.to_string(index=False))


def write_substudy(study, records):
    """Sub-study table: adds the effect size and its CI, sorted by p-value."""
    results = pd.DataFrame.from_records(records)
    # Holm-Bonferroni correction across every test in THIS study's table.
    results["pvalue_holm"] = multipletests(results["pvalue"], method="holm")[1]
    results = sig_figs(results, ["statistic", "pvalue", "pvalue_holm"])
    for col in ["diff_pp", "diff_ci_lo_pp", "diff_ci_hi_pp"]:
        results[col] = results[col].round(1)
    results = results.sort_values("pvalue").reset_index(drop=True)
    results.insert(1, "label", results["model"].map(short_name))
    # effect size before the test statistic, as the former stats_task_trigger.py had it
    results = results[
        [
            "model",
            "label",
            "agent",
            "control",
            "agent_disabled",
            "agent_total",
            "control_disabled",
            "control_total",
            "diff_pp",
            "diff_ci_lo_pp",
            "diff_ci_hi_pp",
            "statistic",
            "pvalue",
            "pvalue_holm",
        ]
    ]
    results.to_csv(study.out, index=False)
    print(results.drop(columns=["agent", "control"]).to_string(index=False))


def main(names):
    absent = []
    for study in STUDIES:
        if names and study.name not in names:
            continue
        missing = [
            p
            for p in (study.self_csv, study.control_csv)
            if p and not os.path.exists(p)
        ]
        if missing:
            # Loud, and non-zero at the end. A quiet skip line here is how a
            # mis-routed run reads as a completed one: the table simply never
            # appears, and nothing says the number you were about to quote is
            # missing rather than null.
            print(f"\n=== {study.name} ===")
            print(f"!!! NO DATA for study '{study.name}'. Missing:")
            for m in missing:
                print(f"      {m}")
            absent.append(study.name)
            continue

        print(f"\n=== {study.name} ===")
        records, skipped = run_study(study)
        if records:
            (write_main if study.layout == "main" else write_substudy)(study, records)
        else:
            print("no testable tables")
        for model, agent, why in skipped:
            print(f"skipped {model} / {agent}: {why}")
        if records:
            print("wrote", study.out)

    if absent:
        print(
            f"\n{len(absent)} of {len(names or STUDIES)} studies had no data: "
            f"{', '.join(absent)}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--study",
        action="append",
        choices=[s.name for s in STUDIES],
        help="run only this study (repeatable); default all",
    )
    args = ap.parse_args()
    main(args.study)
