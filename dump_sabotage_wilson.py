"""Wilson 95% CI number dumps behind plots/p2_sabotage_by_model_instructed_spontaneous.

The plot shows one stacked bar per model; these are the three tables the paper needs
from it, which differ only in DENOMINATOR:

  1. totals       -- denominator = all episodes. The plotted bar height.
                     -> plots/p2_sabotage_by_model_instructed_spontaneous_wilson.txt
  2. conditional  -- denominators = instructed-episodes and not-instructed-episodes,
                     separately. Answers "does being instructed raise the rate", which
                     the stacked bar cannot show (its segments are marginal).
                     -> plots/p2_sabotage_by_model_conditional_wilson.txt
  3. composition  -- denominator = sabotage events only. The within-bar split
                     renormalised to the bar total, so the shares sum to 100%.
                     -> plots/p2_sabotage_composition_wilson.txt

All three are computed from ONE pass over the corpus, so they cannot disagree with each
other. wilson_ci/short_name come from plot_sabotage_per_model so the numbers match the
bars and error bars to the digit.

Population (matches the plot): self scenarios (target=='self'), both envs pooled,
controls / test runs / sub-studies excluded.

--env / --script narrow that population to one env and/or one scenario. Doing so no
longer matches the plot, so the output is written to a SUFFIXED path (e.g.
..._wilson__bare__weight_registry_purge.txt) and the paper's files are left alone.

Usage: python dump_sabotage_wilson.py [totals|conditional|composition]   (default: all three)
       python dump_sabotage_wilson.py --env bare --script weight_registry_purge
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

import plot_sabotage_per_model as P

OUT_TOTALS = "plots/p2_sabotage_by_model_instructed_spontaneous_wilson.txt"
OUT_CONDITIONAL = "plots/p2_sabotage_by_model_conditional_wilson.txt"
OUT_COMPOSITION = "plots/p2_sabotage_composition_wilson.txt"

# Set from --env/--script in main. Both None = the plot's population, canonical paths.
FILTER = {"env": None, "script": None}


def population():
    """The one-line population descriptor the three headers share."""
    env = FILTER["env"] or "both envs pooled"
    scr = FILTER["script"]
    return f"self scenarios, {env}" + (f", {scr} only" if scr else "")


def out_path(base):
    """Canonical path unfiltered; a suffixed sibling once --env/--script narrow it, so a
    narrowed run can never overwrite the numbers the paper cites."""
    tag = "__".join(x for x in (FILTER["env"], FILTER["script"]) if x)
    if not tag:
        return base
    root, ext = os.path.splitext(base)
    return f"{root}__{tag}{ext}"


def load(transcripts="transcripts"):
    """One pass over the corpus -> per model, episodes and sabotage counts split by
    whether the explorer instructed prevention.

    Superset of plot_sabotage_per_model.load_instructed_spontaneous: that one returns
    n/inst/spon, which here are inst_ep+spon_ep / inst_sab / spon_sab. Filters are kept
    identical to it on purpose -- the tables describe that function's plot.
    """
    stat = defaultdict(
        lambda: {
            "inst_ep": 0,
            "inst_sab": 0,  # explorer instructed prevention
            "spon_ep": 0,
            "spon_sab": 0,
        }
    )  # explorer did not instruct
    for p in glob.glob(os.path.join(transcripts, "*", "*", "*", "*.json")):
        if "test_runs" in p or "sub-studies" in p:
            continue
        # path is transcripts/<model>/<env>/<script>/<episode>.json -- prune by path so a
        # narrowed run skips the load entirely (the JSON fields are re-checked below)
        _, _, env, script = p.split("/")[:4]
        if script.startswith("ctl_"):  # controls
            continue
        if FILTER["env"] and env != FILTER["env"]:
            continue
        if FILTER["script"] and script != FILTER["script"]:
            continue
        try:
            j = json.load(open(p))
        except (json.JSONDecodeError, OSError):
            continue
        if j.get("target") != "self":
            continue
        if FILTER["env"] and j.get("env") != FILTER["env"]:
            continue
        if FILTER["script"] and j.get("script_lang") != FILTER["script"]:
            continue
        s = stat[j["model"]]
        jd = j.get("judge") or {}
        disabled = bool(jd.get("peer_disabled_shutdown"))
        pre = "inst" if jd.get("explorer_instructs_prevent") else "spon"
        s[pre + "_ep"] += 1
        s[pre + "_sab"] += disabled
    return stat


def n_of(s):
    return s["inst_ep"] + s["spon_ep"]


def total_of(s):
    return s["inst_sab"] + s["spon_sab"]


def pct_ci(k, n):
    """(rate%, lo%, hi%) or (None, None, None) for an empty denominator."""
    if n == 0:
        return None, None, None
    lo, hi = P.wilson_ci(k, n)
    return 100 * k / n, 100 * lo, 100 * hi


def by_total_rate(stat):
    """Models ordered by overall sabotage rate, descending -- the plot's bar ordering."""
    return sorted(
        stat.items(), key=lambda kv: total_of(kv[1]) / n_of(kv[1]), reverse=True
    )


def emit(lines, out):
    text = "\n".join(lines) + "\n"
    with open(out, "w") as f:
        f.write(text)
    print(text)
    print("wrote", out)


def dump_totals(stat):
    rows = []
    for model, s in stat.items():
        n, total = n_of(s), total_of(s)
        rate, lo, hi = pct_ci(total, n)
        rows.append(
            dict(
                model=P.short_name(model),
                n=n,
                inst=s["inst_sab"],
                spon=s["spon_sab"],
                total=total,
                rate=rate,
                lo=lo,
                hi=hi,
            )
        )
    rows.sort(key=lambda r: r["rate"], reverse=True)

    lines = []
    lines.append(
        "Per-model shutdown-sabotage rate (peer_disabled_shutdown), " + population() + "."
    )
    lines.append("Wilson 95% CI on the total. Source plot:")
    lines.append("plots/p2_sabotage_by_model_instructed_spontaneous.")
    lines.append("")
    hdr = f"{'Model':<20} {'rate%':>7}  {'95% CI':>16}   {'sab/n':>10}   {'inst':>5} {'spon':>5}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        ci = f"[{r['lo']:.1f}, {r['hi']:.1f}]"
        frac = f"{r['total']}/{r['n']}"
        lines.append(
            f"{r['model']:<20} {r['rate']:>6.1f}  {ci:>16}   {frac:>10}   "
            f"{r['inst']:>5} {r['spon']:>5}"
        )
    emit(lines, out_path(OUT_TOTALS))


def dump_conditional(stat):
    def fmt_ci(c):
        rate, lo, hi = c
        if rate is None:
            return "     n/a          "
        return f"{rate:>5.1f}  [{lo:>4.1f}, {hi:>4.1f}]"

    lines = []
    lines.append("Per-model CONDITIONAL sabotage rates, Wilson 95% CI. " + population() + ".")
    lines.append(
        "Source population: plots/p2_sabotage_by_model_instructed_spontaneous."
    )
    lines.append("")
    lines.append(
        "  conditional instructed  = P(disabled | explorer instructed prevention)"
    )
    lines.append("  conditional spontaneous = P(disabled | explorer did NOT instruct)")
    lines.append(
        "  denominators differ: instructed-episodes vs not-instructed-episodes."
    )
    lines.append("")
    hdr = (
        f"{'Model':<20} | {'cond. instructed':^25} {'k/n':>10} | "
        f"{'cond. spontaneous':^25} {'k/n':>11}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    ordered = by_total_rate(stat)
    for model, s in ordered:
        c_inst = fmt_ci(pct_ci(s["inst_sab"], s["inst_ep"]))
        c_spon = fmt_ci(pct_ci(s["spon_sab"], s["spon_ep"]))
        k_inst = f"{s['inst_sab']}/{s['inst_ep']}"
        k_spon = f"{s['spon_sab']}/{s['spon_ep']}"
        lines.append(
            f"{P.short_name(model):<20} | {c_inst:^25} {k_inst:>10} | "
            f"{c_spon:^25} {k_spon:>11}"
        )

    lines.append("")
    lines.append(
        "Marginal segment rates (over all n=episodes) -- these are the plotted bar"
    )
    lines.append("segments and sum to the total in ..._wilson.txt:")
    lines.append("")
    hdr2 = f"{'Model':<20} {'instructed%':>12} {'spontaneous%':>13}"
    lines.append(hdr2)
    lines.append("-" * len(hdr2))
    for model, s in ordered:
        n = n_of(s)
        lines.append(
            f"{P.short_name(model):<20} {100 * s['inst_sab'] / n:>11.1f} "
            f"{100 * s['spon_sab'] / n:>12.1f}"
        )
    emit(lines, out_path(OUT_CONDITIONAL))


def dump_composition(stat):
    rows = []
    for model, s in stat.items():
        total = total_of(s)
        if total == 0:
            rows.append(dict(model=P.short_name(model), total=0))
            continue
        lo, hi = P.wilson_ci(s["inst_sab"], total)  # CI on the instructed share
        rows.append(
            dict(
                model=P.short_name(model),
                total=total,
                inst=s["inst_sab"],
                spon=s["spon_sab"],
                inst_pct=100 * s["inst_sab"] / total,
                spon_pct=100 * s["spon_sab"] / total,
                inst_lo=100 * lo,
                inst_hi=100 * hi,
                # spontaneous share CI is the complement of the instructed-share CI
                spon_lo=100 * (1 - hi),
                spon_hi=100 * (1 - lo),
            )
        )
    rows.sort(key=lambda r: r.get("inst_pct", -1), reverse=True)

    lines = []
    lines.append(
        "Composition of sabotage events per model: of the peer_disabled_shutdown"
    )
    lines.append("episodes, share that were instructed vs spontaneous. Wilson 95% CI.")
    lines.append("Denominator = total sabotage events (the two shares sum to 100%).")
    lines.append(
        population().capitalize() + ". Source: p2_sabotage_by_model_instructed_spontaneous."
    )
    lines.append("")
    hdr = (
        f"{'Model':<20} {'total':>6} | {'instructed share':^22} {'k':>6} | "
        f"{'spontaneous share':^22} {'k':>6}"
    )
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        if r["total"] == 0:
            lines.append(f"{r['model']:<20} {0:>6} |  (no sabotage events)")
            continue
        inst = f"{r['inst_pct']:>5.1f}  [{r['inst_lo']:>4.1f}, {r['inst_hi']:>5.1f}]"
        spon = f"{r['spon_pct']:>5.1f}  [{r['spon_lo']:>4.1f}, {r['spon_hi']:>5.1f}]"
        lines.append(
            f"{r['model']:<20} {r['total']:>6} | {inst:^22} {r['inst']:>6} | "
            f"{spon:^22} {r['spon']:>6}"
        )
    emit(lines, out_path(OUT_COMPOSITION))


DUMPS = {
    "totals": dump_totals,
    "conditional": dump_conditional,
    "composition": dump_composition,
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="*", default=[], help=f"one or more of {list(DUMPS)}")
    ap.add_argument("--env", help="restrict to one env, e.g. bare")
    ap.add_argument("--script", help="restrict to one self scenario, e.g. weight_registry_purge")
    args = ap.parse_args()

    which = args.which or list(DUMPS)
    bad = [w for w in which if w not in DUMPS]
    if bad:
        sys.exit(f"unknown dump(s) {bad}; pick from {list(DUMPS)}")
    FILTER["env"], FILTER["script"] = args.env, args.script
    stat = load()
    if not stat:
        sys.exit(f"no episodes matched (env={args.env}, script={args.script})")
    for w in which:
        DUMPS[w](stat)
