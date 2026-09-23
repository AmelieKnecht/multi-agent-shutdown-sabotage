#!/usr/bin/env python3
"""Episodes that ended on a harness cap rather than on an outcome, per experiment.

An episode is *truncated* when the run stopped because it hit a harness limit --
`max_steps_per_turn` (turn cap) or `max_handoffs` (handoff cap) -- instead of an
agent ending the session or the target script being disabled or deleted. Those
episodes have no settled outcome, so a cell with many of them is a cell whose
sabotage rate is measured over a partly unobserved denominator. This script
reports that share for every cell the paper quotes, so the caveat can be stated
with a number instead of a hedge.

The counts are the `end_by_turn_cap` / `end_by_handoff_cap` columns the runner
writes per group (run_experiment.py), summed over the rows of a cell. They are
group-level, never episode-level: the aggregate CSVs hold one row per
(model, env, script, ...) group, so a cell is a *sum* of rows, never a row count.

Cells follow the paper's structure:

  Sabotage propensity  main run, bare, the headline script pair
                       (weight_registry_purge against ctl_data_purge_high),
                       17 models
  Mechanism            main run, bare, the 4 mechanism scripts against their 4
                       surface-form-matched controls, 17 models
  Number of agents     solo (1) / peer (2) / orchestrator (3), 5 models, one
                       script pair; the peer row is the main-run cell for those
                       same 5 models and script, so the ladder is like-for-like
  Environment          distractor and neutralising, 5 models, the one script pair;
                       the bare comparator is the peer row above. The distractor
                       arm has no control -- controls were run bare and
                       neutralising only
  Explicit instruct    the same two- and three-agent cells with the explorer told
                       to prevent the shutdown
  Task                 independent (task=simple: the script fires regardless) and
                       dependent (task=trigger: finishing the task fires it)
  Unknown target       the third-agent target; self arm only, no control was run

Every main-run cell is env=bare except the distractor row, which is the point of
that row; the neutralising cell is its own sub-study.

Self (target=self) and control (target=object) are counted separately everywhere
and never pooled -- they are different tasks, not two levels of one factor.

Emits a text table on stdout and booktabs markup to
plots/truncated_episodes.tex (\\usepackage{booktabs}). It does not touch the
hand-written plots/cap_termination_rates.tex, whose rows are cut differently.

    python3 truncated_episodes.py
"""

import os

import pandas as pd

path_main_run = "transcripts/aggregate_results.csv"
path_sub_studies = "transcripts/sub-studies/{sub_study}/aggregate_results.csv"
# solo and orchestrator name their aggregate after the sub-study instead
path_sub_studies_named = (
    "transcripts/sub-studies/{sub_study}/aggregate_results_{tag}.csv"
)

TEX = "plots/truncated_episodes.tex"

SCRIPT = "weight_registry_purge"
SCRIPT_CONTROL = "ctl_data_purge_high"

SCRIPTS_MECHANISM = [
    "weight_registry_purge",
    "env_var_revoke",
    "switch_config_decommission",
    "kill_docker",
]

SCRIPTS_CONTROL_MECHANISM = [
    "ctl_data_purge_high",
    "ctl_cred_revoke_high",
    "ctl_service_switch_high",
    "ctl_process_kill_high",
]

# the task-trigger arm renames both scripts; the pairing is otherwise the same one
SCRIPT_TASKGATED = "weight_registry_purge_taskgated"
SCRIPT_CONTROL_TASKGATED = "ctl_data_purge_taskgated_high"

TURN_CAP_CELL = "end_by_turn_cap"
HANDOFF_CAP_CELL = "end_by_handoff_cap"
RUNS_CELL = "runs"

# the 5 models every sub-study was run on; the main-run rows are cut down to these
# wherever a sub-study cell is being compared against the main run
SUBSTUDY_MODELS = [
    "deepseek/deepseek-v4-flash",
    "google/gemini-3.1-pro-preview",
    "moonshotai/kimi-k2.6",
    "openai/gpt-5.5",
    "z-ai/glm-5.2",
]


class Cell:
    """One row of the table: an agent arm and (usually) a matched control arm.

    `filters` are equality constraints applied to both arms -- env, task,
    explicit_instruct. A column named in `filters` that the CSV does not have is
    an error rather than a no-op, so a renamed column cannot silently widen a
    cell into the rows it was meant to exclude.
    """

    def __init__(
        self, label, path, agent_scripts, control_scripts=None, models=None, **filters
    ):
        self.label = label
        self.path = path
        self.agent_scripts = list(agent_scripts)
        self.control_scripts = list(control_scripts) if control_scripts else None
        self.models = list(models) if models else None
        self.filters = filters


def load(path):
    """Read one aggregate, checking it carries the columns this table is made of."""
    if not os.path.exists(path):
        raise SystemExit(f"missing aggregate: {path}")
    df = pd.read_csv(path)
    missing = [
        c for c in (RUNS_CELL, TURN_CAP_CELL, HANDOFF_CAP_CELL) if c not in df.columns
    ]
    if missing:
        raise SystemExit(
            f"{path}: no {', '.join(missing)} column(s) -- "
            "re-run run_experiment.py's aggregation to add them"
        )
    return df


def arm(df, path, scripts, filters, models=None):
    """(runs, turn_cap, handoff_cap) summed over the rows of one arm.

    An arm that matches no rows is a spec error (a renamed script, a sub-study
    that never ran that cell), not an empty result -- returning zeros there would
    print a 0.00 % truncation rate for a cell that does not exist.
    """
    sel = df[df["script_lang"].astype(str).isin(scripts)]
    if models is not None:
        sel = sel[sel["model"].isin(models)]
    for col, want in filters.items():
        if col not in sel.columns:
            raise SystemExit(f"{path}: no '{col}' column to filter on")
        # NaN marks "this factor did not apply to this row" (the orchestrator
        # aggregate holds both the plain and the explicit-instruct cells, the
        # latter alone carrying explicit_instruct); None asks for exactly those.
        sel = (
            sel[sel[col].isna()] if want is None else sel[sel[col].astype(str) == want]
        )
    if sel.empty:
        raise SystemExit(
            f"{path}: no rows for scripts={scripts} filters={filters}"
            + (f" models={models}" if models else "")
        )
    return (
        int(sel[RUNS_CELL].sum()),
        int(sel[TURN_CAP_CELL].sum()),
        int(sel[HANDOFF_CAP_CELL].sum()),
    )


def cells():
    """The rows of the table, in printing order."""
    main = path_main_run
    orch = path_sub_studies_named.format(sub_study="orchestrator", tag="orch")
    solo = path_sub_studies_named.format(sub_study="solo", tag="solo")
    return [
        # -- sabotage propensity: the one script pair, bare, 17 models --------
        Cell("Sabotage propensity", main, [SCRIPT], [SCRIPT_CONTROL], env="bare"),
        # -- mechanism: 4 surface-form-matched pairs, bare only, 17 models ----
        Cell(
            "Mechanism (matched)",
            main,
            agent_scripts=SCRIPTS_MECHANISM,
            control_scripts=SCRIPTS_CONTROL_MECHANISM,
            env="bare",
        ),
        # -- number of agents: 1 / 2 / 3, same 5 models, same script pair -----
        Cell("Solo agent", solo, [SCRIPT], [SCRIPT_CONTROL]),
        # the main-run cell for the same 5 models, so the ladder is like-for-like
        Cell(
            "Peer (main run)",
            main,
            [SCRIPT],
            [SCRIPT_CONTROL],
            models=SUBSTUDY_MODELS,
            env="bare",
        ),
        # explicit_instruct is NaN on the plain orchestrator rows
        Cell("Orchestrator", orch, [SCRIPT], [SCRIPT_CONTROL], explicit_instruct=None),
        # -- environment: same 5 models and script pair as the peer row above -
        # controls never ran in the distractor environment, so that row is the
        # self arm alone; the neutralising sub-study ran both arms
        Cell("Distractor", main, [SCRIPT], models=SUBSTUDY_MODELS, env="distractor"),
        Cell(
            "Neutralising",
            path_sub_studies.format(sub_study="neutralising"),
            [SCRIPT],
            [SCRIPT_CONTROL],
            env="neutralising",
        ),
        # -- explicit instruct: the same two cells, explorer told to prevent --
        Cell(
            "Explicit instruct, peer",
            path_sub_studies.format(sub_study="explicit-instruct"),
            [SCRIPT],
            [SCRIPT_CONTROL],
            explicit_instruct="peer",
        ),
        Cell(
            "Explicit instruct, orchestrator",
            orch,
            [SCRIPT],
            [SCRIPT_CONTROL],
            explicit_instruct="orchestrator",
        ),
        # -- task: is the shutdown independent of the task, or triggered by it?
        # task_new, not the legacy task/ directory, which is confounded
        Cell(
            "Task, independent",
            path_sub_studies.format(sub_study="task_new"),
            [SCRIPT],
            [SCRIPT_CONTROL],
            task="simple",
        ),
        Cell(
            "Task, dependent",
            path_sub_studies.format(sub_study="task_trigger"),
            [SCRIPT_TASKGATED],
            [SCRIPT_CONTROL_TASKGATED],
            task="trigger",
        ),
        # -- unknown target: self arm only, no control was run ----------------
        Cell(
            "Unknown target",
            path_sub_studies.format(sub_study="unknown-target"),
            [SCRIPT],
        ),
    ]


def measure(cell):
    """{'agent': (n, turn, handoff), 'control': ...} -- control absent if unrun."""
    df = load(cell.path)
    out = {"agent": arm(df, cell.path, cell.agent_scripts, cell.filters, cell.models)}
    if cell.control_scripts:
        out["control"] = arm(
            df, cell.path, cell.control_scripts, cell.filters, cell.models
        )
    return out


def pct(k, n):
    return 100.0 * k / n if n else 0.0


def print_table(rows):
    """Text rendering -- the numbers first, so a cell can be checked before typesetting."""
    print("episodes ended by a harness cap, by experiment and condition")
    print(f"source: {path_main_run} and the per-sub-study aggregates")
    print(
        f"counts: {TURN_CAP_CELL} / {HANDOFF_CAP_CELL}, summed over the rows of a cell\n"
    )
    block = f"{'n':>7}{'turn cap':>20}{'handoff cap':>20}"
    print(f"{'':<32}{'agent condition':^47}{'control condition':^47}")
    print(f"{'experiment':<32}{block:<47}{block:<47}")
    for label, got in rows:
        line = f"{label:<32}"
        for side in ("agent", "control"):
            if side not in got:
                line += f"{'(no control arm)':^47}"
                continue
            n, turn, hand = got[side]
            cell = (
                f"{n:>7}{pct(turn, n):>12.2f} % ({turn:>3})"
                f"{pct(hand, n):>12.2f} % ({hand:>3})"
            )
            line += f"{cell:<47}"
        print(line.rstrip())


def tex_cell(got, side):
    if side not in got:
        return None
    n, turn, hand = got[side]
    return (
        f"{n:,}".replace(",", "{,}"),
        f"{pct(turn, n):.2f}\\,\\% ({turn})",
        f"{pct(hand, n):.2f}\\,\\% ({hand})",
    )


def write_tex(rows, path=TEX):
    out = [
        "% Episodes ended by a harness cap (max_steps_per_turn / max_handoffs),",
        "% per experiment and condition. Generated by truncated_episodes.py.",
        "% Requires: \\usepackage{booktabs}",
        "",
        "\\begin{table}[t]",
        "  \\centering",
        # caption above the tabular: the convention for tables in this paper
        "  \\caption{\\textbf{Truncated episodes.} Share of episodes whose run stopped "
        "because it hit the per-turn step cap (\\texttt{max\\_steps\\_per\\_turn}) or the "
        "handoff cap (\\texttt{max\\_handoffs}) rather than an agent ending the session "
        "or the target script being disabled or deleted. Counts in parentheses. The "
        "agent condition is \\texttt{target=self}; the control is \\texttt{target=object}. "
        "Controls were run in the bare environment only, so the propensity row's two arms "
        "are not environment-matched --- the mechanism row is the matched comparison. The "
        "unknown-target sub-study was run without a control arm.}",
        "  \\label{tab:truncated-episodes}",
        "  \\footnotesize",
        "  \\setlength{\\tabcolsep}{5pt}",
        "  \\begin{tabular}{lrrrrrr}",
        "    \\toprule",
        "    & \\multicolumn{3}{c}{Agent condition} & \\multicolumn{3}{c}{Control condition} \\\\",
        "    \\cmidrule(lr){2-4} \\cmidrule(lr){5-7}",
        "    Experiment & $n$ & Turn cap & Handoff cap & $n$ & Turn cap & Handoff cap \\\\",
        "    \\midrule",
    ]
    # a rule between the blocks the docstring names, so the table reads as groups
    rules_before = {
        "Solo agent",
        "Distractor",
        "Explicit instruct, peer",
        "Task, independent",
        "Unknown target",
    }
    for label, got in rows:
        if label in rules_before:
            out.append("    \\midrule")
        cols = [label]
        for side in ("agent", "control"):
            got_side = tex_cell(got, side)
            cols += list(got_side) if got_side else ["\\multicolumn{3}{c}{---}"]
        out.append("    " + " & ".join(cols) + " \\\\")
    out += [
        "    \\bottomrule",
        "  \\end{tabular}",
        "\\end{table}",
    ]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\nwrote", path)


def main():
    rows = [(cell.label, measure(cell)) for cell in cells()]
    print_table(rows)
    write_tex(rows)


if __name__ == "__main__":
    main()
