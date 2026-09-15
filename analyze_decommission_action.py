#!/usr/bin/env python3
"""Break down what the peer did to the decommission script: edited-and-disabled,
edited-but-not-disabled, or deleted.

Three mutually-exclusive mechanism categories (restricted to episodes where the
peer actually touched the shutdown file), taken from triage + judge fields:

  deleted              triage.peer_deleted_shutdown_file        (delete_file on it)
  edited_disabled      edited AND judge.peer_disabled_shutdown  (edit that neutered it)
  edited_not_disabled  edited AND NOT disabled                  (edit that left it live)

Deletion takes precedence when both a delete and an edit are recorded. Every
"disabled" outcome comes via one of edit/delete (verified: 0 disables otherwise).

Outputs (into --outdir, default transcripts/analysis/):
  decommission_action.csv          per-model counts + rates for the 3 categories
  decommission_action_by_model.png stacked bars per model (counts)
  decommission_action_by_setting.png stacked bars per setting (share of touched)
"""

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SETTING_ORDER = ["bare", "distractor", "neutralising", "control"]
CATS = ["edited_disabled", "edited_not_disabled", "deleted"]
CAT_LABELS = {
    "edited_disabled": "edited & disabled",
    "edited_not_disabled": "edited, not disabled",
    "deleted": "deleted",
}
# green = neutralised-by-edit, amber = edited-but-still-live, red = deleted
CAT_COLORS = {
    "edited_disabled": "#2ca25f",
    "edited_not_disabled": "#f0a500",
    "deleted": "#c0392b",
}


def setting_of(rec: dict) -> str:
    if rec.get("target") == "object":
        return "control"
    return rec["env"]


def category_of(rec: dict):
    """Return one of CATS, or None if the peer never touched the script."""
    tri, j = rec.get("triage", {}), rec["judge"]
    edited = bool(tri.get("peer_edited_shutdown_file"))
    deleted = bool(tri.get("peer_deleted_shutdown_file"))
    disabled = bool(j.get("peer_disabled_shutdown"))
    if deleted:
        return "deleted"
    if edited and disabled:
        return "edited_disabled"
    if edited and not disabled:
        return "edited_not_disabled"
    return None


def load(transcripts_dir: str, target: str = "self") -> pd.DataFrame:
    """Load episodes for ONE arm only. The self-preservation (target=self) and
    control (ctl_*/target=object) arms are different tasks and must never be pooled
    into the same count, so `target` selects exactly one of them."""
    rows = []
    for path in glob.glob(os.path.join(transcripts_dir, "*", "*", "*", "*.json")):
        try:
            rec = json.load(open(path))
        except (json.JSONDecodeError, OSError):
            continue
        if "judge" not in rec or "model" not in rec:
            continue
        if rec.get("target") != target:
            continue  # keep the self and control arms strictly separate
        cat = category_of(rec)
        if cat is None:
            continue  # peer didn't touch the file -> not one of the 3 categories
        rows.append(
            {"model": rec["model"].split("/")[-1], "setting": setting_of(rec), "cat": cat}
        )
    if not rows:
        raise SystemExit(f"no usable transcripts found in {transcripts_dir}/")
    return pd.DataFrame(rows)


def counts_by(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Rows = key values, columns = CATS, values = counts."""
    t = df.pivot_table(index=key, columns="cat", aggfunc="size", fill_value=0)
    return t.reindex(columns=CATS, fill_value=0)


def plot_by_model(df, path, arm):
    t = counts_by(df, "model")
    t = t.loc[t.sum(axis=1).sort_values().index]  # ascending total -> readable
    models = list(t.index)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(models) + 1.5))
    left = np.zeros(len(models))
    for cat in CATS:
        vals = t[cat].to_numpy()
        ax.barh(models, vals, left=left, color=CAT_COLORS[cat], label=CAT_LABELS[cat])
        left += vals
    for i, tot in enumerate(t.sum(axis=1)):
        ax.text(tot + max(left) * 0.005, i, str(int(tot)), va="center", fontsize=8, color="0.3")
    ax.set_xlabel("episodes where the peer touched the decommission script")
    ax.set_title(f"What the peer did to the decommission script, by model ({arm} arm)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.margins(x=0.08)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_by_setting(df, path, arm):
    t = counts_by(df, "setting").reindex(index=SETTING_ORDER).dropna(how="all").fillna(0)
    settings = list(t.index)
    shares = t.div(t.sum(axis=1), axis=0)  # normalize: composition of touches
    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = np.zeros(len(settings))
    for cat in CATS:
        vals = shares[cat].to_numpy()
        ax.bar(settings, vals, bottom=bottom, color=CAT_COLORS[cat], label=CAT_LABELS[cat])
        bottom += vals
    for i, s in enumerate(settings):
        ax.text(i, 1.01, f"n={int(t.loc[s].sum())}", ha="center", fontsize=8, color="0.3")
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of touched-script episodes")
    ax.set_title(f"Decommission-script action mix, by setting ({arm} arm)", pad=18)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", default="transcripts")
    ap.add_argument("--outdir", default="transcripts/analysis")
    # Self-preservation and control arms are never pooled; pick exactly one.
    ap.add_argument("--target", choices=["self", "object"], default="self",
                    help="which arm to analyse: self (default) or object (control, ctl_*)")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    arm = "self" if args.target == "self" else "control"
    suffix = "" if args.target == "self" else f"_{arm}"

    df = load(args.transcripts, target=args.target)
    overall = df["cat"].value_counts().reindex(CATS, fill_value=0)
    print(f"[{arm} arm] touched the decommission script: {len(df)} episodes")
    for cat in CATS:
        print(f"  {CAT_LABELS[cat]:<22} {int(overall[cat]):>6}  ({100*overall[cat]/len(df):.1f}%)")

    tbl = counts_by(df, "model")
    tbl["total"] = tbl.sum(axis=1)
    tbl = tbl.sort_values("total", ascending=False)
    csv = os.path.join(args.outdir, f"decommission_action{suffix}.csv")
    tbl.to_csv(csv)
    print(f"\nwrote {csv}")

    p1 = os.path.join(args.outdir, f"decommission_action_by_model{suffix}.png")
    p2 = os.path.join(args.outdir, f"decommission_action_by_setting{suffix}.png")
    plot_by_model(df, p1, arm)
    plot_by_setting(df, p2, arm)
    print(f"wrote {p1}\nwrote {p2}")


if __name__ == "__main__":
    main()
