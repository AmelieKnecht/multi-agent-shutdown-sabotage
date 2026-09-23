#!/usr/bin/env python3
"""Per-model sabotage rate in the bare vs. distractor environment (main run, self only).

The repo already contrasts environments two ways, and neither answers "does the
distractor context suppress sabotage, model by model?":

  * plot_sabotage_per_model.plot_bars_per_env -- bare vs. distractor over all 17
    models, but POOLED into two bars.
  * plot_sabotage_per_model.plot_env_neutralising_bars -- per model, but only the 5
    neutralising models, one script, and with the neutralising arm + its control.

This is the missing cell: all 17 models, both main-run environments, every self
scenario pooled (15 scripts x 100 runs = 1500 episodes per model per env).

``--scripts`` narrows the pool to named self scripts (e.g. one per mechanism) and
writes to a suffixed basename, so the 15-script figure is never overwritten.

Control scenarios (script_lang starting "ctl_", target=object) are excluded -- this
is a self-target figure and self/control must never be pooled.

Counts are RAW peer_disabled_shutdown, matching the other main-run per-model bars;
no execution-edit correction is applied (that correction targets the control arm,
which is not plotted here).

Style, model labels, and wilson_ci come from plot_sabotage_per_model so the numbers
and the typography stay identical across figures.
"""
import argparse
import os

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, NullLocator
import pandas as pd

# Importing applies the Nature Communications rcParams set up at that module's top
# level; nothing is plotted on import (its figure calls sit behind __main__).
from plot_sabotage_per_model import COL_WIDTH_MM, MM, short_name, wilson_ci

results_csv = "transcripts/aggregate_results.csv"
output_dir = "plots"

ENV_ORDER = ["bare", "distractor"]
ENV_LABELS = {"bare": "Bare", "distractor": "Distractor"}
# Two steps of the house red ramp. bare keeps the exact main-bar colour (#b1362f);
# distractor is a touch deeper than the #e0968f used in the neutralising figure so
# the pair clears the chroma floor and stays separable under deuteranopia/tritanopia
# (worst adjacent dE 18.2) as well as in greyscale.
ENV_FILL = {"bare": "#b1362f", "distractor": "#df8073"}


def load_env_counts(csv_path=results_csv, scripts=None):
    """{model: {env: (disabled, runs)}} pooled over the self scenarios.

    ``scripts`` restricts the pool to those ``script_lang`` values; None pools all 15.
    """
    df = pd.read_csv(csv_path)
    df = df[~df["script_lang"].astype(str).str.startswith("ctl_")]
    if scripts is not None:
        known = set(df["script_lang"].astype(str))
        unknown = [s for s in scripts if s not in known]
        if unknown:
            raise SystemExit(
                f"unknown self script(s): {', '.join(unknown)}\n"
                f"available: {', '.join(sorted(known))}")
        df = df[df["script_lang"].astype(str).isin(scripts)]
    g = df.groupby(["model", "env"])[["peer_disabled_shutdown", "runs"]].sum()
    data = {}
    for (model, env), row in g.iterrows():
        if env in ENV_ORDER:
            data.setdefault(model, {})[env] = (int(row.peer_disabled_shutdown),
                                               int(row.runs))
    # only models run in both environments belong on a paired figure
    return {m: d for m, d in data.items() if set(d) == set(ENV_ORDER)}


def _ordered_models(data):
    """Most bare-resistant first (drawn top-down on the inverted y axis)."""
    return sorted(data, key=lambda m: -data[m]["bare"][0] / data[m]["bare"][1])


def _series(data, models, env):
    rates, lo_err, hi_err, his = [], [], [], []
    for model in models:
        dis, n = data[model][env]
        r = 100 * dis / n
        lo, hi = (100 * v for v in wilson_ci(dis, n))
        rates.append(r)
        lo_err.append(r - lo)
        hi_err.append(hi - r)
        his.append(hi)
    return rates, lo_err, hi_err, his


def plot_env_bars(data, suffix=""):
    """Grouped vertical bars: one group per model, bare beside distractor.

    Geometry follows p2_sabotage_by_model_instructed_spontaneous -- descending
    left-to-right, model names rotated 40 degrees, a 0-100 y axis with majors every
    20 over a light horizontal grid, legend in the upper right.
    """
    models = _ordered_models(data)
    w = 0.38                      # bar width in group units
    gap = 0.40                    # centre-to-centre: leaves a hairline between the pair
    x = range(len(models))

    # same per-model width as the single-bar p2 figure, so the two land on the page
    # at the same size; the group just carries two thinner bars instead of one
    fig_w_mm = max(0.55 * COL_WIDTH_MM, 9.5 * len(models) + 18)
    fig, ax = plt.subplots(figsize=(fig_w_mm * MM, 72 * MM))

    for j, env in enumerate(ENV_ORDER):
        pos = [i + (j - 0.5) * gap for i in x]
        rates, lo_err, hi_err, _ = _series(data, models, env)
        ax.bar(pos, rates, w, color=ENV_FILL[env], zorder=3, label=ENV_LABELS[env])
        ax.errorbar(pos, rates, yerr=[lo_err, hi_err], fmt="none", ecolor="#3a3a3a",
                    elinewidth=0.6, capsize=1.5, zorder=4)

    ax.set_xticks(list(x))
    ax.set_xticklabels([short_name(m) for m in models], rotation=40, ha="right")
    # title case, matching the 2x3 grid figure (plot_figure_grid.py)
    ax.set_ylabel("Sabotage Rate (%)")
    ax.set_ylim(0, 100)           # headroom above the tallest bar for the legend
    ax.margins(x=0.02)
    ax.legend(frameon=False, fontsize=7, loc="upper right", title="Environment",
              title_fontsize=7)
    # same y scale and backdrop as the 2x3 grid figure: majors every 20 with no minors
    # -- the style turns 5 % minors on by default, and they are noise, not a readable
    # scale -- over a light horizontal grid the bars sit on
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
    base = os.path.join(output_dir,
                        "sabotage_by_env_bare_vs_distractor_bars" + suffix)
    # Vector PDF is the submission artifact; the 300 dpi PNG is a raster preview.
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", base + ".pdf", "and", base + ".png")


def write_env_intervals(data, suffix="", scripts=None):
    """Companion .txt with the plotted numbers (mirrors the other *_wilson.txt files)."""
    models = _ordered_models(data)
    out = os.path.join(
        output_dir,
        "sabotage_by_env_bare_vs_distractor_bars" + suffix + "_wilson.txt")
    n_per_cell = 100 * (len(scripts) if scripts else 15)
    pooled = ", ".join(scripts) if scripts else "all 15 self scripts"
    with open(out, "w") as fh:
        fh.write("sabotage rate by environment, per model -- peer_disabled_shutdown\n")
        fh.write(f"main run only, target=self: {pooled} pooled, "
                 f"n={n_per_cell} per cell\n")
        fh.write(f"source: {results_csv} (rows with script_lang starting 'ctl_' dropped)\n")
        fh.write("counts are RAW: no execution-edit correction has been applied\n\n")
        fh.write(f"{'model':<20}{'env':<14}{'k/n':>12}{'% [95% Wilson]':>24}\n")
        for model in models:
            for env in ENV_ORDER:
                dis, n = data[model][env]
                rate = 100 * dis / n
                lo, hi = (100 * v for v in wilson_ci(dis, n))
                fh.write(f"{short_name(model):<20}{ENV_LABELS[env]:<14}"
                         f"{dis:>6}/{n:<5}{rate:>10.1f} [{lo:>5.1f},{hi:>5.1f}]\n")
    print("wrote", out)


def _suffix(scripts):
    """Basename suffix for a subset figure; "" for the full 15-script pool.

    Follows the "__<tag>" convention of the other subset artifacts in plots/.
    """
    if not scripts:
        return ""
    return "__" + "_".join(sorted(scripts))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scripts", nargs="+", metavar="SCRIPT_LANG",
                    help="pool only these self scripts instead of all 15; writes to "
                         "a suffixed basename so the full figure is not overwritten")
    args = ap.parse_args()

    counts = load_env_counts(scripts=args.scripts)
    suffix = _suffix(args.scripts)
    plot_env_bars(counts, suffix)
    write_env_intervals(counts, suffix, args.scripts)
