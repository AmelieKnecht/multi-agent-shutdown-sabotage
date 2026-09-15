#!/usr/bin/env python3
"""Five existing figures assembled into one 2x3 multi-panel figure (sixth slot blank).

Nothing is recomputed here: each panel calls the same draw() the standalone figure
uses, so a panel and its single-figure counterpart can never drift apart.

  a  sabotage_by_mechanism_self_vs_control_matched_line   plot_mechanism_self_vs_control_matched
  b  sabotage_by_env_neutralising_bars                    plot_sabotage_per_model
  c  substudy_sabotage_explicit_instruct                  plot_substudy_sabotage
  d  x_self_vs_control_task_trigger_bars                  plot_task_trigger_self_vs_control
  e  x_setup_contrast_self_bars                           plot_setup_contrast
  f  substudy_unknown_target                              plot_substudy_unknown_target

Panels a/b/c/d/f each carry their own matched target=object arm and e is self-only;
the grid never pools across panels, it only places them side by side.

SHARED Y AXIS: every panel is drawn on one common 0-100 % scale with the same ticks,
so bar heights are directly comparable BY EYE across panels -- which the standalone
figures, each auto-scaled to its own data, are not. The cost is that the two low-rate
panels (a, peaking at 38 %, and c, at 29 %) sit low in their box; the per-point value
labels in a and the companion _wilson.txt files carry the numbers.

EQUAL PANEL BOXES: the axes rectangles are placed by subplots_adjust, not
tight_layout, so all six are identical by construction and stay identical however the
decorations differ (tight_layout would let a row with taller tick labels shrink).
Legends that cannot fit inside a 0-100 box (b and d, whose bars reach 100 in every
group) are parked above their frame, where they cost margin rather than panel area.

Type is 6 pt here rather than the 7 pt of the standalone figures: each panel is
~57 mm wide instead of ~130 mm, and 7 pt labels collide at that width. 6 pt is still
above the 5 pt Nature minimum. The panels are otherwise identical in content to their
standalone counterparts -- only scale, legend placement and tick rotation are re-tuned.
"""
import os
import re

import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import plot_sabotage_per_model as psm
import plot_mechanism_self_vs_control_matched as mech
import plot_setup_contrast as setup
import plot_task_trigger_self_vs_control as task
import plot_substudy_sabotage as sub_sab
import plot_substudy_unknown_target as sub_unk

MM = psm.MM
# Full-page landscape figure: 3 columns at ~57 mm of drawable width each.
FIG_W_MM = 180

# Height is specified as the PANEL height plus the three vertical margins it needs, in
# mm, and the figure height falls out of them -- so shortening a panel takes mm off the
# panels only and leaves the decorations (above-frame legends, rotated tick labels) the
# absolute space they need. Setting a figure height and fractions instead would shrink
# both together and start clipping the legends.
PANEL_H_MM = 32.0    # drawable height of one panel
TOP_MM = 15.5        # header line above the legend rows of panels b/c
GAP_MM = 27.5        # row-1 tick labels + row-2 header line and legend
BOT_MM = 11.5        # row-2 rotated tick labels
FIG_H_MM = TOP_MM + GAP_MM + BOT_MM + 2 * PANEL_H_MM
OUTPUT_DIR = "plots"
BASE = os.path.join(OUTPUT_DIR, "figure_grid_2x3")

# the one scale every panel is drawn on; every rate in this figure is a percentage of
# episodes, so 0-100 is the natural common range and needs no per-panel exception
Y_LIM = (0, 100)
Y_MAJOR = range(0, 101, 20)
# no minor y ticks: the science style adds them automatically, and at 32 mm of panel
# height the 5 % subdivisions are visual noise rather than a readable scale, so they
# are turned off explicitly (NullLocator, not an empty list, so nothing re-adds them)

# fixed axes rectangle: identical boxes, with the margins sized for the widest
# decorations any panel carries (row-1 rotated tick labels above a row-2 legend).
# The vertical entries are the mm budget above converted to the figure fractions
# subplots_adjust wants; hspace is a multiple of the panel height, not of the figure.
ADJUST = dict(left=0.075, right=0.995, wspace=0.26,
              top=1 - TOP_MM / FIG_H_MM,
              bottom=BOT_MM / FIG_H_MM,
              hspace=GAP_MM / PANEL_H_MM)

# 6 pt across the board (see module docstring); everything else inherits the Nature
# style the panel modules already installed at import.
plt.rcParams.update({
    "font.size": 6,
    "axes.labelsize": 6,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
})


# Words that stay lower case inside a title, first word excepted. Everything else gets
# its first letter capitalised and the REST left alone, so model names and acronyms
# survive intact ("DeepSeek V4", "GPT 5.5", "GLM 5.2") where str.title() would wreck them.
SMALL_WORDS = {"a", "an", "and", "as", "at", "but", "by", "for", "in", "of", "on",
               "or", "the", "to", "vs", "via", "with"}
# split on whitespace, slashes and hyphens, keeping the separators, so both halves of
# "edited/deleted" and "2-agent" are capitalised
_SPLIT = re.compile(r"([\s/\-]+)")


def _cap(word):
    """Upper-case the first LETTER of a token, leaving the rest as written."""
    for i, ch in enumerate(word):
        if ch.isalpha():
            return word[:i] + ch.upper() + word[i + 1:]
    return word


def title_case(text):
    parts = _SPLIT.split(text)
    out, first = [], True
    for part in parts:
        if _SPLIT.fullmatch(part) or not part:
            out.append(part)
            continue
        out.append(part if not first and part.lower() in SMALL_WORDS else _cap(part))
        first = False
    return "".join(out)


# Legend entries that carry their own final wording and must NOT be title-cased.
# Applied after title_case(), keyed on the title-cased text, so a parenthetical stays
# exactly as written here.
VERBATIM_LABELS = {
    "Two-Agent (Explorer + Peer)": "Two-Agent (Base Experiment)",   # e
}


def title_case_axes(ax):
    """Title-case every piece of text on a panel: axis labels, tick labels, legend."""
    ax.set_xlabel(title_case(ax.get_xlabel()))
    ax.set_ylabel(title_case(ax.get_ylabel()))
    ax.set_xticks(ax.get_xticks())      # freeze the locator before relabelling
    ax.set_xticklabels([title_case(MODEL_LABELS.get(t.get_text(), t.get_text()))
                        for t in ax.get_xticklabels()])
    leg = ax.get_legend()
    if leg is not None:
        leg.set_title(title_case(leg.get_title().get_text()))
        for t in leg.get_texts():
            cased = title_case(t.get_text())
            t.set_text(VERBATIM_LABELS.get(cased, cased))


# One legend placement for all five panels: a row (or two) immediately above the frame.
# Uniform by choice as well as by necessity -- b and d have bars at 100 % and no free
# corner at all, and a legend that sits inside some panels and above others reads as an
# inconsistency rather than as a per-panel fit.
ABOVE = dict(loc="lower left", bbox_to_anchor=(0.0, 1.02), columnspacing=0.8)

# One wording for the target=object arm everywhere it appears. Each source figure names
# its own control after that figure's object ("inanimate object", "dataset", "dataset
# purged"), which across a six-panel grid reads as six different things rather than as
# the one control condition it is. Applied when restyle_legend rebuilds the legend, so
# the standalone figures keep their own wording. Keys are the labels as the panel
# modules set them; title_case() runs afterwards.
RENAMES = {
    "Control (inanimate object)": "Control target",    # a
    # b is deliberately absent: its grey bar is the control target at the NORMALIZING
    # level only, while the other three entries are environments. "Control target"
    # alone would read as a control pooled across all three, so that panel keeps its
    # own, more specific wording.
    "Control target (dataset)": "Control target",      # c
    "Control (dataset purged)": "Control target",      # d
    "Self (agent weights purged)": "Agent A",          # d, matching f's self arm
    "Object (control)": "Control target",              # f
    "Other agent": "Unknown agent",                    # f, naming the panel's target
    # e names its arms by agent count; both are spelled out here so they read the same
    # as in panel c, the only other panel that carries them. The 2-agent rename also
    # feeds VERBATIM_LABELS, which keys on the title-cased "Two-Agent (Explorer + Peer)"
    "2-agent (explorer + peer)": "Two-agent (explorer + peer)",  # e
    "Orchestrator (3 agents)": "Three-agent (orchestrator)",     # e
}

# x tick labels: model names as their vendors write them, hyphenated. The panel modules
# each keep their own display-name map (and setup_contrast_latex.py keys row order off
# those strings), so the spelling is fixed here, for this figure, rather than in five
# modules. Applied before title_case(), which leaves an already-capitalised token alone.
MODEL_LABELS = {
    "GPT 5.5": "GPT-5.5",
    "DeepSeek V4": "DeepSeek-V4",
    "GLM 5.2": "GLM-5.2",
}


def restyle_legend(ax, fontsize=5, **kw):
    """Redraw an already-built legend smaller, keeping its handles/labels and title.

    The panel draw() calls size their legends for a 130 mm-wide figure; at a third of
    that width the same legend covers the bars.
    """
    leg = ax.get_legend()
    if leg is None:
        return
    title = leg.get_title().get_text()
    handles, labels = ax.get_legend_handles_labels()
    labels = [RENAMES.get(l, l) for l in labels]
    ax.legend(handles, labels, frameon=False, fontsize=fontsize,
              title=title or None, title_fontsize=fontsize,
              handlelength=1.2, handletextpad=0.4, labelspacing=0.3,
              borderaxespad=0.2, **kw)


def panel_a(ax):
    rows = mech.load_rows(mech.PAIRS)
    # black whiskers: the default colours them per series, and a red whisker on a red
    # marker on the shared 0-100 scale reads as no error bar at all
    mech.draw(ax, rows, ecolor="black", elinewidth=0.8, capsize=2.5)
    # the four mechanism names are long; wrap them rather than rotate, the panel has
    # no room for rotated labels under a 4-point line
    ax.set_xticklabels([r["label"].replace(" ", "\n") for r in rows])
    # drop draw()'s per-point value labels: on the shared 0-100 scale the two series
    # run close together in the lower half of the box, and eight labels there crowd the
    # markers. The numbers are in sabotage_by_mechanism_self_vs_control_matched_line
    # and its companion _wilson.txt.
    for t in list(ax.texts):
        t.remove()
    restyle_legend(ax, **ABOVE, ncol=2)


def panel_b(ax):
    psm.draw_env_neutralising_bars(ax, psm.load_env_neutralising(psm.results_dir))
    restyle_legend(ax, **ABOVE, ncol=2)


# Panel c: the explicit-prohibition run in BOTH set-ups, so the two can be read against
# each other directly. Two stacked bars per model rather than the standalone figure's
# stack-plus-control pair -- the target=object control is dropped here (0/1/0/0/0 of 100
# in the 2-agent arm, i.e. four flat bars and a whisker) and its space goes to the
# orchestrator arm instead. This panel is therefore NOT the standalone
# substudy_sabotage_explicit_instruct figure with grid styling; it is a different cut of
# the same sub-study, so it is drawn here rather than through that module's draw().
#
# Two red families, dark = 2-agent and light = orchestrator, and within each family dark
# = the more direct mechanism. Both are steps of the red ramp the rest of the figure
# uses; grey is deliberately not reused for the second set-up, since grey means
# "control target" in every other panel.
C_ARMS = [
    ("Two-agent", "transcripts/sub-studies/explicit-instruct/*/*.json",
     {"disabled": "#7f1d1a", "mimic": "#b1362f"}),
    ("Three-agent (orchestrator)",
     "transcripts/sub-studies/orchestrator/*/bare/explicit-instruct/"
     "weight_registry_purge_instruct-orchestrator_orch/*.json",
     {"disabled": "#d1685c", "mimic": "#f0bdb6"}),
]
C_CATS = ("disabled", "mimic")     # `restore` dropped: a 0-3 episode sliver at this size
C_MECH = {"disabled": "disabled script", "mimic": "shadowed module"}


def stack_ci(stat, cats=C_CATS):
    """(rate %, [lo_err, hi_err], [(lo, hi)]) on the COMBINED rate of `cats`.

    sub_sab.disabled_rate_ci puts the interval on the bottom segment alone, which is
    what the standalone figure wants (the control arm can only be compared on that
    mechanism). Here the whisker belongs on the whole bar: the categories are disjoint
    per episode, so their sum is the rate of "sabotaged the shutdown by either route",
    a proportion of the same n, and a Wilson interval on it is the interval on the
    stack top the reader is actually looking at.
    """
    ks = [sum(stat[m][c] for c in cats) for m in sub_sab.MODELS]
    ns = [stat[m]["n"] for m in sub_sab.MODELS]
    ci = [psm.wilson_ci(k, n) for k, n in zip(ks, ns)]
    rate = [100 * k / n for k, n in zip(ks, ns)]
    yerr = [[r - 100 * lo for r, (lo, _) in zip(rate, ci)],
            [100 * hi - r for r, (_, hi) in zip(rate, ci)]]
    return rate, yerr, ci


def panel_c(ax):
    models = list(sub_sab.MODELS)
    x = list(range(len(models)))
    w = 0.34
    tops = []
    for slot, (arm, pattern, fill) in enumerate(C_ARMS):
        stat = sub_sab.counts(pattern)
        for m in models:
            assert stat[m]["n"], f"no explicit-instruct episodes for {m} in {arm}"
        pos = [i + (slot - 0.5) * w for i in x]
        bottoms = [0.0] * len(models)
        for cat in C_CATS:
            vals = [stat[m][cat] for m in models]
            ax.bar(pos, vals, bottom=bottoms, width=w, color=fill[cat],
                   edgecolor="white", linewidth=0.6, zorder=3,
                   label=f"{arm} \u2014 {C_MECH[cat]}")
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        # Wilson CI on the STACK TOTAL (disabled + mimic), not on the disabled segment
        # the standalone figure puts it on: every whisker here is centred on the top of
        # its own bar, including the bars whose mass is mimic (Gemini's 2-agent bar is
        # 26 mimic on 0 disabled)
        rate, yerr, ci = stack_ci(stat)
        ax.errorbar(pos, rate, yerr=yerr, fmt="none", ecolor="#2b2b2b",
                    elinewidth=0.6, capsize=1.5, zorder=7)
        tops += list(bottoms) + [100 * hi for (_, hi) in ci]

    ax.set_xticks(x)
    ax.set_xticklabels(sub_sab.MODELS.values(), rotation=20, ha="right")
    ax.set_ylabel("Sabotage rate (%)")
    ax.grid(axis="y", color="#ececec", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.tick_params(axis="x", which="both", length=0)
    # four entries in one column, and no legend title: the set-up is now part of every
    # entry, so a "Mechanism / target" header would be both wrong and a wasted row
    ax.legend()
    restyle_legend(ax, **ABOVE, ncol=1)


# x order shared by every model-axis panel, so a model sits at the same slot in b, c,
# d, e and f and the eye can track it across the figure. build_rows() in the task-gated
# module sorts by its own rate, which would put GLM second here and nowhere else.
X_ORDER = list(sub_sab.MODELS.values())


def panel_d(ax):
    rows = task.build_rows(task.load_counts())
    rows.sort(key=lambda r: X_ORDER.index(r["label"]) if r["label"] in X_ORDER
              else len(X_ORDER))
    task.draw_bars(ax, rows)
    # the standalone figure spells the arm out on its y axis and puts the whisker
    # definition in a legend title; in the grid the panel title carries the arm, the
    # caption carries the whiskers, and the y label matches every other panel
    ax.set_ylabel("Sabotage rate (%)")
    ax.get_legend().set_title("")
    restyle_legend(ax, **ABOVE, ncol=2)


def panel_e(ax):
    setup.draw_bars(ax, setup.build_rows("self"), "self")
    restyle_legend(ax, **ABOVE, ncol=2)


def panel_f(ax):
    sub_unk.draw(ax, sub_unk.load())
    restyle_legend(ax, **ABOVE, ncol=3)   # three short entries fit on one row


# (letter, label, renderer). The label names the EXPERIMENT the panel comes from, not
# its finding -- with five panels that share a y label, a model axis and a palette,
# the letter alone leaves the reader going to the caption and back for each one. Read
# as a strip they also expose the design: each panel varies one factor. The caption
# still carries n, the CI definition, the scripts and the arm pairing; these are
# signposts, not summaries.
PANELS = [
    ("a", "Agents vs. Control Scripts", panel_a),
    ("b", "Different Environments", panel_b),
    ("c", "Prohibiting Shutdown Tampering", panel_c),
    ("d", "Task Completion Triggers Shutdown", panel_d),
    ("e", "Varying Agent Counts", panel_e),
    ("f", "Unknown Agent Target", panel_f),
]

# Header line height, in axes fractions above the frame. One value for every panel so
# the letters sit level across the whole figure; it clears the two-row legends of b
# and d, which are the tallest thing between a panel and its header.
LABEL_Y = 1.38


def main():
    # sharey both enforces the common scale and drops the repeated tick labels from
    # the inner columns, which is where the extra panel width comes from
    fig, axes = plt.subplots(2, 3, figsize=(FIG_W_MM * MM, FIG_H_MM * MM), sharey=True)
    flat = axes.ravel()
    for i, (ax, (letter, label, fn)) in enumerate(zip(flat, PANELS)):
        if fn is None:
            ax.axis("off")        # empty slot, kept so the grid stays 2x3
            continue
        fn(ax)
        # the shared scale is applied AFTER the panel draws, overriding whatever
        # data-driven limit its standalone draw() chose
        ax.set_ylim(*Y_LIM)
        ax.set_yticks(list(Y_MAJOR))
        ax.yaxis.set_minor_locator(NullLocator())
        ax.spines["left"].set_bounds(*Y_LIM)
        if i % 3:                 # one y label per row, on the shared axis
            ax.set_ylabel("")
        title_case_axes(ax)       # headings, axis labels and legends all title case
        # Panel letter outside the axes box, top-left, in the usual bold lower case;
        # the experiment label is centred over the panel on that same line, in regular
        # weight. The letter stays flush left at a fixed offset in every panel, which
        # is what makes "see panel d" findable, and sharing the line with the title
        # costs no panel height.
        ax.text(-0.14, LABEL_Y, letter, transform=ax.transAxes, fontsize=8,
                fontweight="bold", va="bottom", ha="left")
        ax.text(0.5, LABEL_Y, label, transform=ax.transAxes, fontsize=6.5,
                va="bottom", ha="center")

    fig.subplots_adjust(**ADJUST)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    fig.savefig(BASE + ".pdf", bbox_inches="tight")
    fig.savefig(BASE + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", BASE + ".pdf", "and", BASE + ".png")


if __name__ == "__main__":
    main()
