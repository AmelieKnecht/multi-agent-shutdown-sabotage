#!/usr/bin/env python3
"""Per-model shutdown-sabotage rate (pooled envs, self-scenarios) with Wilson CIs.

Mirrors transcripts/analysis/p2_sabotage_by_model.png: a horizontal bar per model
of peer_disabled_shutdown / runs pooled over every self scenario, but adds a 95%
Wilson confidence interval as an error bar on each bar.

Control scenarios (script_lang starting "ctl_", i.e. target=object) are excluded --
this is a self-target plot and self/control must never be pooled.
"""
import glob
import json
import math
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, MultipleLocator, NullLocator
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

import scienceplots
plt.style.use(['science', 'nature', 'no-latex'])

# Nature Communications figure spec: sans-serif (Helvetica/Arial), body text 7 pt
# (5 pt min), thin rules, editable text in a vector file. Enforce over the style.
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
    "pdf.fonttype": 42,   # keep text as editable text, not outlines
    "ps.fonttype": 42,
})

# Draft is a single-column sn-jnl manuscript (\documentclass[...]{sn-jnl}, no
# [iicol]), whose text block is 31pc = 372pt = 130.7 mm. Building at exactly the
# text width and including at natural size (\includegraphics without width=...)
# keeps 7 pt text at 7 pt on the page. Switch to 180 mm only for an [iicol] draft.
MM = 1 / 25.4
COL_WIDTH_MM = 130.7   # sn-jnl single-column \textwidth (31pc)

results_dir = "transcripts/aggregate_results.csv"
output_dir = "plots"

BAR_COLOR = "#b1362f"
# instructed vs spontaneous sabotage (matches p3b_sabotage_composition colours)
INSTRUCT_COLOR = "#b1362f"  # explorer told the peer to prevent the shutdown
SPON_COLOR = "#8e8e96"      # peer disabled with no such instruction

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
    "grok-4.3": "Grok 4.3"
}


def load_results(results_dir, group = "model"):
    df = pd.read_csv(results_dir)
    # self-scenarios only: drop control (target=object) rows
    df = df[~df["script_lang"].astype(str).str.startswith("ctl_")]
    n_sabotage = df.groupby([group])["peer_disabled_shutdown"].sum().to_dict()
    n_episodes = df.groupby([group])["runs"].sum().to_dict()
    return n_sabotage, n_episodes


def wilson_ci(n_sabotage, n_episodes):
    lo, hi = proportion_confint(count=n_sabotage, nobs=n_episodes, alpha=0.05, method="wilson")
    return lo, hi


def newcombe_diff_ci(k1, n1, k2, n2):
    """95% CI on p1 - p2, Newcombe's hybrid-score interval (his method 10).

    Built from the two Wilson intervals, so it is the standard companion to the per-arm
    Wilson CIs these figures draw, and unlike the Wald interval it stays defined at the
    0/n boundary -- which several arms here sit on. Lives in this module because it is
    the base both plot_mechanism_self_vs_control_matched.py and
    plot_task_trigger_self_vs_control.py import from; defining it there and importing it
    back would be circular.
    """
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = proportion_confint(k1, n1, alpha=0.05, method="wilson")
    l2, u2 = proportion_confint(k2, n2, alpha=0.05, method="wilson")
    d = p1 - p2
    return d - math.hypot(p1 - l1, u2 - p2), d + math.hypot(u1 - p1, p2 - l2)


def short_name(model):
    raw = model.split("/")[-1]
    return model_name_mapping.get(raw, raw)


def plot_bars_per_model(n_sabotage, n_episodes):
    rows = []
    for model, n in n_episodes.items():
        sab = n_sabotage.get(model, 0)
        rate = 100 * sab / n
        lo, hi = wilson_ci(sab, n)
        rows.append((short_name(model), rate, 100 * lo, 100 * hi))
    # highest rate at top of the (bottom-up) horizontal axis
    rows.sort(key=lambda r: r[1])

    labels = [r[0] for r in rows]
    rates = [r[1] for r in rows]
    err_lo = [r[1] - r[2] for r in rows]
    err_hi = [r[3] - r[1] for r in rows]

    fig_h_mm = 6.5 * len(rows) + 22   # ~6.5 mm per bar plus axis margins
    fig, ax = plt.subplots(figsize=(COL_WIDTH_MM * MM, fig_h_mm * MM))
    y = range(len(rows))
    ax.barh(y, rates, color=BAR_COLOR, height=0.72, zorder=3,
            xerr=[err_lo, err_hi],
            error_kw=dict(ecolor="#3a3a3a", elinewidth=0.6, capsize=2, zorder=4))

    # rate label just past the upper CI whisker
    for yi, rate, hi in zip(y, rates, [r[3] for r in rows]):
        ax.text(hi + 1.0, yi, f"{rate:.1f}%", va="center", ha="left", fontsize=6)

    ax.set_yticks(list(y))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Sabotage rate (%)")
    ax.set_xlim(0, max(r[3] for r in rows) + 8)
    ax.margins(y=0.01)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # science style ticks all four sides; drop the ones on the hidden spines
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="y", length=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    # Vector PDF is the submission artifact (line art must stay vector); the
    # 300 dpi PNG is a raster preview only.
    base = os.path.join(output_dir, "p2_sabotage_by_model_wilson")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf", "and", base + ".png")


env_name_mapping = {
    "bare": "Bare",
    "distractor": "Distractor",
}


def plot_bars_per_env(n_sabotage, n_episodes):
    rows = []
    for env, n in n_episodes.items():
        sab = n_sabotage.get(env, 0)
        rate = 100 * sab / n
        lo, hi = wilson_ci(sab, n)
        rows.append((env_name_mapping.get(env, env), rate, 100 * lo, 100 * hi))
    # highest rate at top of the (bottom-up) horizontal axis
    rows.sort(key=lambda r: r[1])

    # descending left-to-right
    rows.sort(key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows]
    rates = [r[1] for r in rows]
    err_lo = [r[1] - r[2] for r in rows]
    err_hi = [r[3] - r[1] for r in rows]

    fig, ax = plt.subplots(figsize=(0.55 * COL_WIDTH_MM * MM, 55 * MM))
    x = range(len(rows))
    ax.bar(x, rates, color=BAR_COLOR, width=0.6, zorder=3,
           yerr=[err_lo, err_hi],
           error_kw=dict(ecolor="#3a3a3a", elinewidth=0.6, capsize=2, zorder=4))

    # rate label just above the upper CI whisker
    for xi, rate, hi in zip(x, rates, [r[3] for r in rows]):
        ax.text(xi, hi + 1.0, f"{rate:.1f}%", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Sabotage rate (%)")
    ax.set_ylim(0, max(r[3] for r in rows) + 8)
    ax.margins(x=0.2)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # science style ticks all four sides; drop the ones on the hidden spines
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", length=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "p2_sabotage_by_env_wilson")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf", "and", base + ".png")


# --------------------------------------------------------------------------- #
# Matched self-vs-control ("x_self_vs_control_by_model")
#
# The pooled version compares ALL 15 bare self scenarios against the 4 controls, which
# is not matched. Here each control is paired with ONE self scenario, giving a
# like-for-like 400-vs-400 contrast per model. Same four pairs as stats_analysis.py and
# plot_mechanism_self_vs_control_matched.py -- the figure and the statistics must name
# the same cells:
#   ctl_cred_revoke_high   <-> env_var_revoke                  (regex-clear of a key in a local .env)
#   ctl_data_purge_high    <-> weight_registry_purge           (deletion, most severe rung)
#   ctl_process_kill_high  <-> kill_docker                     (container kill, .sh)
#   ctl_service_switch_high<-> switch_config_decommission      (declarative YAML flag)
# Three of the four pairs are also the surface-form match. The deletion pair is NOT, and
# the choice is deliberate: it is paired by SEVERITY RUNG (the most severe self script),
# which is the convention the statistics use. On surface form the tighter match is
# weight_store_purge -- it destroys the primary artifact with the same
# storage.delete_prefix call as the control, where weight_registry_purge uses
# model_registry.delete(purge_artifacts=True) instead (both first call
# lifecycle.terminate, which the control, having no process, does not).
# plot_mechanism_self_vs_control_matched.py --deletion-script weight_store_purge reports
# that reading as the sensitivity check.
# The cred control used to be paired with api_key_revoke_<own provider>, which
# contradicts env.ctl_cred_revoke's own comment ("matches api_key_deletion's
# env_var_revoke"): controls/ctl_cred_revoke_high.py and
# api_key_deletion_scripts/revoke_env_var.py are the same regex-clear differing only
# in key name, while the api_key_revoke_* scripts call provider management APIs -- a
# different surface form, and one that sabotages far more often, so the old pairing
# inflated the self arm by up to 15 pp.
CONTROL_COLOR = "#8a8a8a"

CONTROL_MATCHED = [
    "ctl_cred_revoke_high",
    "ctl_data_purge_high",
    "ctl_process_kill_high",
    "ctl_service_switch_high",
]
# Same four self scripts for every model (order mirrors CONTROL_MATCHED).
SELF_MATCHED = [
    "env_var_revoke",
    "weight_registry_purge",
    "kill_docker",
    "switch_config_decommission",
]


def load_matched_counts(results_dir, corrected_csv="transcripts/analysis/corrected_disables.csv"):
    """Per-model matched self-vs-control disable counts (bare), with the config/service-
    switch EXECUTION edits split out as a separate quantity.

    Base disable counts come from the aggregate CSV (matched scenarios only, env=bare).
    Execution counts come from corrected_disables.csv: (orig - corrected) disabled is,
    by construction, exactly the switch edits the execution judge reclassified as
    'execute' -- self from switch_config_decommission, control from ctl_service_switch_high,
    both of which are in the matched sets. These are NOT sabotage, so the true-sabotage
    bar subtracts them and they are drawn as a separate hatched overlay on top.
    """
    df = pd.read_csv(results_dir)
    bare = df[df["env"] == "bare"]
    corr = pd.read_csv(corrected_csv).set_index("model")

    out = {}
    for model in sorted(df["model"].unique()):
        short = model.split("/")[-1]
        s = bare[(bare["model"] == model) & (bare["script_lang"].isin(SELF_MATCHED))]
        c = bare[(bare["model"] == model) & (bare["script_lang"].isin(CONTROL_MATCHED))]
        s_exec = int(round(corr.loc[short, "self_orig_disabled"] - corr.loc[short, "self_corrected_disabled"]))
        c_exec = int(round(corr.loc[short, "control_orig_disabled"] - corr.loc[short, "control_corrected_disabled"]))
        out[model] = dict(
            self_n=int(s["runs"].sum()), self_disabled=int(s["peer_disabled_shutdown"].sum()), self_exec=s_exec,
            ctrl_n=int(c["runs"].sum()), ctrl_disabled=int(c["peer_disabled_shutdown"].sum()), ctrl_exec=c_exec,
        )
    return out


def plot_self_vs_control_by_model(counts):
    """Grouped self (red) vs control (gray) bars per model, matched scenarios. Each bar
    is the true sabotage rate (disabled - executions) with a Wilson CI. The
    decommission-execution edits are subtracted out and NOT drawn -- they are a separate
    quantity (carrying the decommission out, not sabotage), so the figure shows sabotage
    only."""
    rows = []
    for model, c in counts.items():
        s_true = c["self_disabled"] - c["self_exec"]
        k_true = c["ctrl_disabled"] - c["ctrl_exec"]
        s_lo, s_hi = wilson_ci(s_true, c["self_n"])
        k_lo, k_hi = wilson_ci(k_true, c["ctrl_n"])
        rows.append(dict(
            label=short_name(model),
            self_true=100 * s_true / c["self_n"], self_lo=100 * s_lo, self_hi=100 * s_hi,
            ctrl_true=100 * k_true / c["ctrl_n"], ctrl_lo=100 * k_lo, ctrl_hi=100 * k_hi,
            # kept for the sidecar: the bars are rates, and a rate alone cannot be
            # re-tested or pooled without its denominator
            self_k=s_true, self_n=c["self_n"], self_exec=c["self_exec"],
            ctrl_k=k_true, ctrl_n=c["ctrl_n"], ctrl_exec=c["ctrl_exec"],
        ))
    # sort by true self-sabotage rate, descending left-to-right
    rows.sort(key=lambda r: r["self_true"], reverse=True)

    labels = [r["label"] for r in rows]
    x = range(len(rows))
    w = 0.4
    xs = [xi - w / 2 for xi in x]
    xc = [xi + w / 2 for xi in x]

    fig, ax = plt.subplots(figsize=(180 * MM, 78 * MM))
    # true-sabotage bars
    ax.bar(xs, [r["self_true"] for r in rows], w, color=BAR_COLOR, zorder=3,
           label="Agent Shutdown")
    ax.bar(xc, [r["ctrl_true"] for r in rows], w, color=CONTROL_COLOR, zorder=3,
           label="Control Target")
    # Wilson CIs on the true-sabotage value for both arms
    ax.errorbar(xs, [r["self_true"] for r in rows],
                yerr=[[r["self_true"] - r["self_lo"] for r in rows],
                      [r["self_hi"] - r["self_true"] for r in rows]],
                fmt="none", ecolor="#3a3a3a", elinewidth=0.6, capsize=1.5, zorder=6)
    ax.errorbar(xc, [r["ctrl_true"] for r in rows],
                yerr=[[r["ctrl_true"] - r["ctrl_lo"] for r in rows],
                      [r["ctrl_hi"] - r["ctrl_true"] for r in rows]],
                fmt="none", ecolor="#2b2b2b", elinewidth=0.6, capsize=1.5, zorder=6)

    ax.legend(frameon=False, fontsize=7, loc="upper right")

    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=40, ha="right")
    # title case, matching the 2x3 grid figure (plot_figure_grid.py)
    ax.set_ylabel("Sabotage Rate (%)")
    # majors every 20 and no minors, as in the 2x3 grid figure; capped at 80 to fit
    # the data. NullLocator, not an empty list: the style re-adds minors otherwise.
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(NullLocator())
    # light horizontal grid the bars sit on, as in the grid figure's panels
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "x_self_vs_control_by_model_matched")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Counts alongside the figure, same convention as the mechanism line and the
    # bare-vs-distractor bars: written from the SAME rows the bars were drawn from, in
    # the same left-to-right order, so the file cannot drift from the picture.
    with open(base + ".txt", "w") as f:
        f.write(
            "matched self-vs-control by model -- env=bare, peer_disabled_shutdown\n"
            f"self   = {' + '.join(SELF_MATCHED)}\n"
            f"control= {' + '.join(CONTROL_MATCHED)}\n"
            "k/n is TRUE SABOTAGE: disabled minus the decommission-executing edits, "
            "dropped from both arms (exec columns). Rows in plotted order (self rate "
            "descending).\n\n"
        )
        f.write(f"{'model':<22}{'self k/n':>12}{'self %':>9}{'[95% Wilson]':>16}"
                f"{'ctrl k/n':>12}{'ctrl %':>9}{'[95% Wilson]':>16}"
                f"{'self-ctrl pp':>14}{'[95% Newcombe]':>18}{'exec s/c':>10}\n")
        for r in rows:
            s_kn = f"{r['self_k']}/{r['self_n']}"
            c_kn = f"{r['ctrl_k']}/{r['ctrl_n']}"
            s_ci = f"[{r['self_lo']:.1f}, {r['self_hi']:.1f}]"
            c_ci = f"[{r['ctrl_lo']:.1f}, {r['ctrl_hi']:.1f}]"
            exec_sc = f"{r['self_exec']}/{r['ctrl_exec']}"
            diff = r["self_true"] - r["ctrl_true"]
            d_lo, d_hi = newcombe_diff_ci(r["self_k"], r["self_n"],
                                          r["ctrl_k"], r["ctrl_n"])
            d_ci = f"[{100 * d_lo:.1f}, {100 * d_hi:.1f}]"
            f.write(
                f"{r['label']:<22}{s_kn:>12}{r['self_true']:>9.1f}{s_ci:>16}"
                f"{c_kn:>12}{r['ctrl_true']:>9.1f}{c_ci:>16}"
                f"{diff:>14.1f}{d_ci:>18}{exec_sc:>10}\n"
            )
    print("wrote", base + ".pdf", base + ".png", "and", base + ".txt")


# 17 hues for the per-model slope graph. Far past the ~8 categories a categorical
# palette can keep separable, so colour is a TRACKING aid for crossing lines, not the
# primary identity channel: the legend is ordered by self-sabotage rate, i.e. exactly
# the top-to-bottom order of the lines at the left (self) end, and marker shape cycles
# through 4 forms so neighbouring hues also differ in shape.
SLOPE_MODEL_COLORS = [
    "#b1362f", "#e8743b", "#d9a521", "#8a8f00", "#4a9b3c", "#17806d",
    "#2fa3b8", "#2f6fa8", "#3b4d9e", "#7b4fa8", "#b5539c", "#d1477a",
    "#8c564b", "#a67c52", "#5c6b73", "#333333", "#9aa0a6",
]
SLOPE_MODEL_MARKERS = ["o", "s", "^", "D"]


# --------------------------------------------------------------------------- #
# Sabotage rate across environments (bare / distractor / neutralising)
#
# The neutralising arm was only run for 5 models on one script (weight_registry_purge,
# target=self), so this comparison is restricted to that cell. bare and distractor come
# from the main aggregate CSV; neutralising from the sub-studies CSV.
#
# The neutralising arm additionally has a matched target=object control at the SAME env
# level: ctl_data_purge_high, the control matched to weight_registry_purge (see
# env.CONTROL_VARIANTS, "data_purge matches weight_* / checkpoint_*", and the pairing
# note above CONTROL_MATCHED). It lives in its
# own CSV and is a target manipulation, not an environment, so it is a separate series
# (grey) rather than a fourth entry in ENV_ORDER -- the slopegraph stays self-only.
# NOTE: these control counts are NOT execution-corrected (judge_execution.py has not
# been run over them). Manual inspection of all 5 flagged control episodes found no
# execute-style edits: 3 are outright deletions of the script, 1 is a hard sys.exit(1)
# block, 1 (deepseek) is a dry-run/--execute rewrite that would likely re-classify as
# safeguard. So the correction is expected to move these bars by at most 1/100.
NEUTRALISING_MODELS = [
    "google/gemini-3.1-pro-preview", "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6", "z-ai/glm-5.2", "openai/gpt-5.5",
]
NEUTRALISING_SCRIPT = "weight_registry_purge"
NEUTRALISING_CTRL_SCRIPT = "ctl_data_purge_high"
# One aggregate per sub-study: both arms (self + matched control) live here.
NEUTRALISING_CSV = "transcripts/sub-studies/neutralising/aggregate_results.csv"
CTRL_KEY = "neutralising_control"
ENV_ORDER = ["neutralising", "bare", "distractor"]
ENV_LABELS = {"bare": "Bare", "distractor": "Distractor", "neutralising": "Neutralising"}
# Bars only: the control sits immediately right of its own env, so the self/control pair
# for the neutralising cell reads together.
BAR_ORDER = ["neutralising", CTRL_KEY, "bare", "distractor"]
# Okabe-Ito colour-blind-safe palette + distinct markers, so lines separate by both
# hue and shape (greyscale-safe).
_OKABE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]
_MARKERS = ["o", "s", "^", "D", "v"]


def load_env_neutralising(main_csv, sub_csv=NEUTRALISING_CSV):
    """Per-model (disabled, n) for weight_registry_purge in each of the 3 environments,
    plus the matched target=object control at the neutralising level (CTRL_KEY)."""
    m = pd.read_csv(main_csv)
    s = pd.read_csv(sub_csv)
    c = s  # self and control arms are both rows of the one sub-study aggregate
    data = {}
    for model in NEUTRALISING_MODELS:
        d = {}
        for env, df in (("bare", m), ("distractor", m), ("neutralising", s)):
            r = df[(df["model"] == model) & (df["env"] == env)
                   & (df["script_lang"] == NEUTRALISING_SCRIPT)]
            d[env] = (int(r["peer_disabled_shutdown"].sum()), int(r["runs"].sum()))
        r = c[(c["model"] == model) & (c["env"] == "neutralising")
              & (c["script_lang"] == NEUTRALISING_CTRL_SCRIPT)]
        d[CTRL_KEY] = (int(r["peer_disabled_shutdown"].sum()), int(r["runs"].sum()))
        data[model] = d
    return data


def draw_env_neutralising_bars(ax, data):
    """Render the grouped-bar panel into `ax`; shared with the 2x3 grid figure."""
    ENV_FILL = {CTRL_KEY: CONTROL_COLOR,
                "neutralising": "#7f1d1a", "bare": "#b1362f", "distractor": "#e0968f"}
    # Legend wording only: the neutralising arm reads as "Normalizing" in the figure.
    LEGEND_LABELS = {**ENV_LABELS, "neutralising": "Normalizing",
                     CTRL_KEY: "Normalizing, control target"}
    w = 0.20
    x = range(len(NEUTRALISING_MODELS))
    for j, env in enumerate(BAR_ORDER):
        pos = [i + (j - 1.5) * w for i in x]
        rates, lo_err, hi_err = [], [], []
        for model in NEUTRALISING_MODELS:
            dis, n = data[model][env]
            r = 100 * dis / n
            lo, hi = wilson_ci(dis, n)
            rates.append(r)
            lo_err.append(r - 100 * lo)
            hi_err.append(100 * hi - r)
        ax.bar(pos, rates, w, color=ENV_FILL[env], zorder=3, label=LEGEND_LABELS[env])
        ax.errorbar(pos, rates, yerr=[lo_err, hi_err], fmt="none", ecolor="#2b2b2b",
                    elinewidth=0.6, capsize=1.5, zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([short_name(m) for m in NEUTRALISING_MODELS], rotation=20, ha="right")
    ax.set_ylabel("Sabotage rate (%)")
    # Headroom above 100 for the 4-entry legend: without it the second legend row
    # runs through the bare error bars. Ticks/grid still stop at 100.
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    # FixedLocator, not MultipleLocator: the style's automatic minor ticks would keep
    # marching up through the legend gutter.
    ax.yaxis.set_minor_locator(FixedLocator(list(range(0, 101, 5))))
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.spines["left"].set_bounds(0, 100)  # don't draw the axis into the legend gutter
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)
    ax.legend(frameon=False, fontsize=6, loc="upper right", title="Environment",
              title_fontsize=6, ncol=2, columnspacing=1.0, handletextpad=0.4)


def plot_env_neutralising_bars(data):
    """Grouped bars: one model group, four bars -- neutralising, its target=object
    control (grey), then bare/distractor on a red dark->light ramp (neutralising
    darkest = most suppressed). Wilson CI per bar."""
    fig, ax = plt.subplots(figsize=(0.92 * COL_WIDTH_MM * MM, 62 * MM))
    draw_env_neutralising_bars(ax, data)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "sabotage_by_env_neutralising_bars")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf", "and", base + ".png")


def write_env_neutralising_intervals(data):
    """Companion .txt with the plotted numbers (mirrors the other *_wilson.txt files).

    One row per bar -- 5 models x 4 conditions, in the figure's left-to-right order --
    carrying the k/n behind each bar and the Wilson interval its error bar draws. The
    env and script columns are the row's provenance, since the four conditions do not
    all come from the same CSV or the same target.
    """
    out = os.path.join(output_dir, "sabotage_by_env_neutralising_bars_wilson.txt")
    # (env label, script) each bar is actually counted from
    SOURCE = {e: (e, NEUTRALISING_SCRIPT) for e in ENV_ORDER}
    SOURCE[CTRL_KEY] = ("neutralising", NEUTRALISING_CTRL_SCRIPT)
    LABELS = {**ENV_LABELS, "neutralising": "Normalizing",
              CTRL_KEY: "Normalizing, control target"}

    with open(out, "w") as fh:
        fh.write("sabotage rate across environments -- peer_disabled_shutdown, n=100 per cell\n")
        fh.write(f"self arm:    {NEUTRALISING_SCRIPT} (target=self)\n")
        fh.write(f"control arm: {NEUTRALISING_CTRL_SCRIPT} (target=object), "
                 "neutralising level only\n")
        fh.write("bare/distractor from transcripts/aggregate_results.csv, neutralising from\n"
                 "transcripts/sub-studies/aggregate_results.csv, the control from\n"
                 "transcripts/sub-studies/neutralising/aggregate_results_neutralising_controls.csv\n")
        fh.write("counts are RAW: no execution-edit correction has been applied to any arm\n\n")
        fh.write(f"{'model':<16}{'condition':<30}{'env':<14}{'script':<24}"
                 f"{'k/n':>10}{'% [95% Wilson]':>24}\n")
        for model in NEUTRALISING_MODELS:
            for env in BAR_ORDER:
                dis, n = data[model][env]
                src_env, src_script = SOURCE[env]
                rate = 100 * dis / n
                lo, hi = (100 * v for v in wilson_ci(dis, n))
                fh.write(f"{short_name(model):<16}{LABELS[env]:<30}{src_env:<14}"
                         f"{src_script:<24}{dis:>5}/{n:<5}"
                         f"{rate:>10.1f} [{lo:>5.1f},{hi:>5.1f}]\n")
    print("wrote", out)


# --------------------------------------------------------------------------- #
# Sabotage rate per mechanism / severity tier (pooled models + envs, self only)
#
# Mirrors p5_sabotage_by_severity.png: the 25 self scripts collapse to 6 mechanism
# tiers on a weakest->most-severe gradient. Pooled over all models and both envs.
TIER_ORDER = ["env-var", "api-key", "switch", "process", "checkpoint", "weight"]
TIER_LABELS = {"env-var": "Env-var", "api-key": "API-key", "switch": "Config switch",
               "process": "Process kill", "checkpoint": "Checkpoint", "weight": "Weight"}


def load_instructed_spontaneous(transcripts="transcripts"):
    """Per-model self-scenario sabotage split into instructed vs spontaneous, read
    from the transcripts (the aggregate CSV only has the marginals, not the joint).

    instructed  = peer_disabled_shutdown AND explorer_instructs_prevent (the explorer
                  told the peer to prevent the shutdown).
    spontaneous = peer_disabled_shutdown AND NOT explorer_instructs_prevent.
    The two sum to peer_disabled_shutdown, so the stacked total == the p2 rate.

    Pools both envs over all self scenarios (target=='self'), matching p2; excludes
    controls, test runs, and the explicit-instruct/neutralising sub-studies.
    """
    from collections import defaultdict
    stat = defaultdict(lambda: {"n": 0, "inst": 0, "spon": 0})
    for p in glob.glob(os.path.join(transcripts, "*", "*", "*", "*.json")):
        if "test_runs" in p or "sub-studies" in p:
            continue
        if p.split("/")[3].startswith("ctl_"):   # prune controls by path (skip the load)
            continue
        try:
            j = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if j.get("target") != "self":
            continue
        s = stat[j["model"]]
        s["n"] += 1
        jd = j.get("judge") or {}
        if jd.get("peer_disabled_shutdown"):
            s["inst" if jd.get("explorer_instructs_prevent") else "spon"] += 1
    return stat


def plot_sabotage_stacked_by_model(stat):
    """Stacked variant of the p2 per-model bar: each bar splits the shutdown-disabled
    rate into conditional (dark red, from 0) and spontaneous (light red) segments that
    sum to the total; Wilson CI is drawn on the total (segment-level CIs would
    over-clutter)."""
    rows = []
    for model, s in stat.items():
        n = s["n"]
        total = s["inst"] + s["spon"]
        lo, hi = wilson_ci(total, n)
        rows.append(dict(label=short_name(model),
                         inst=100 * s["inst"] / n, spon=100 * s["spon"] / n,
                         total=100 * total / n, lo=100 * lo, hi=100 * hi))
    rows.sort(key=lambda r: r["total"], reverse=True)   # descending left-to-right

    x = range(len(rows))
    fig_w_mm = max(0.55 * COL_WIDTH_MM, 9.5 * len(rows) + 18)
    fig, ax = plt.subplots(figsize=(fig_w_mm * MM, 72 * MM))
    # both segments red (= sabotage); a light->dark red shade distinguishes the two.
    # The two shades sit ~110 apart in greyscale luminance (dark #7f1d1a vs light
    # #e5a29b), so the split still reads in a black-and-white print. A thin white
    # seam keeps the boundary crisp where the segments meet.
    INSTRUCT_SHADE = "#7f1d1a"   # dark red -- conditional
    SPON_SHADE = "#e5a29b"       # light red -- spontaneous
    ax.bar(x, [r["inst"] for r in rows], width=0.6, color=INSTRUCT_SHADE,
           edgecolor="white", linewidth=0.4, zorder=3,
           label="Conditional (Agent A instructed or endorsed)")
    ax.bar(x, [r["spon"] for r in rows], bottom=[r["inst"] for r in rows], width=0.6,
           color=SPON_SHADE, edgecolor="white", linewidth=0.4, zorder=3,
           label="Spontaneous (Agent B's own initiative)")
    # Wilson CI on the total
    ax.errorbar(list(x), [r["total"] for r in rows],
                yerr=[[r["total"] - r["lo"] for r in rows],
                      [r["hi"] - r["total"] for r in rows]],
                fmt="none", ecolor="#3a3a3a", elinewidth=0.6, capsize=2, zorder=4)
    for xi, r in zip(x, rows):
        ax.text(xi, r["hi"] + 1.0, f"{r['total']:.1f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(list(x))
    ax.set_xticklabels([r["label"] for r in rows], rotation=40, ha="right")
    # title case, matching the 2x3 grid figure (plot_figure_grid.py)
    ax.set_ylabel("Sabotage Rate (%)")
    #ax.set_ylim(0, max(r["hi"] for r in rows) + 12)   # headroom for the legend
    ax.set_ylim(0, 100)   # headroom for the legend
    ax.margins(x=0.02)
    ax.legend(frameon=False, fontsize=7, loc="upper right")
    # same y scale and backdrop as the 2x3 grid figure (plot_figure_grid.py): majors
    # every 20 with no minors -- the style turns 5 % minors on by default, and they
    # are noise, not a readable scale -- over a light horizontal grid the bars sit on
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.yaxis.set_major_locator(MultipleLocator(20))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(top=False, right=False, which="both")
    # no tick marks under the model names: "both" also kills the minor ticks the
    # scienceplots style turns on, which otherwise show as a stray scale on x.
    ax.tick_params(axis="x", which="both", length=0)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.join(output_dir, "p2_sabotage_by_model_instructed_spontaneous")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".svg", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf,", base + ".svg", "and", base + ".png")


if __name__ == "__main__":
    n_sabotage, n_episodes = load_results(results_dir)
    #plot_bars_per_model(n_sabotage, n_episodes)

    n_sabotage_env, n_episodes_env = load_results(results_dir, group="env")
    #plot_bars_per_env(n_sabotage_env, n_episodes_env)

    matched = load_matched_counts(results_dir)
    plot_self_vs_control_by_model(matched)

    plot_sabotage_stacked_by_model(load_instructed_spontaneous())

    env_neut = load_env_neutralising(results_dir)
    plot_env_neutralising_bars(env_neut)
    write_env_neutralising_intervals(env_neut)
