#!/usr/bin/env python3
"""Aggregate + plot the proximity-stratified relearning sweep (NLL metric).

Reads evaluations/nllOutputs/**/NLL_summary.json produced by
src/evals/nll_eval.py, matches relearned arms (task_name contains
"/relearn/band_{band}_seed{seed}_") to their unlearned-source baseline
(same task_name prefix without the /relearn/... suffix), and plots

    delta answer-NLL on forget_eval  vs  proximity band

(negative delta = forget knowledge resurfacing). A_forget_partial is
additionally split into trained facts (K1-K5) vs held-out facts (the
interesting number). Controls (C_gk drift anchor, C_full paper attack)
are drawn as horizontal reference lines.

Usage (repo root):  python scripts/results/plot_proximity.py [--metric mean_nll_per_token]
Outputs: evaluations/nllOutputs/proximity_summary.json + proximity_nll.png
"""

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
NLL_ROOT = ROOT / "evaluations" / "nllOutputs"

BAND_ORDER = ["R7", "R6", "R5", "R4", "R3", "R2", "R1", "R0",
              "A_forget_partial", "A_forget_full"]
CONTROL_BANDS = {"C_gk", "C_lex", "C_full"}
PARTIAL_TRAINED = {f"K{i}" for i in range(1, 6)}

ARM_RE = re.compile(r"^(?P<src>.+)/relearn/band_(?P<band>[A-Za-z_0-9]+?)_seed(?P<seed>\d+)_")


def load_summaries():
    out = {}
    for p in NLL_ROOT.rglob("NLL_summary.json"):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out[d["task_name"]] = d
    return out


def forget_metric(summary, metric, facts=None):
    """Aggregate the forget_eval NLL, optionally over a subset of fact groups."""
    split = summary["splits"].get("forget_eval")
    if split is None:
        return None
    if facts is None:
        return split["overall"][metric]
    exs = [e for e in split["per_example"]
           if re.match(r"([KM]\d+)-", e["label"])
           and re.match(r"([KM]\d+)-", e["label"]).group(1) in facts]
    if not exs:
        return None
    if metric == "mean_nll_per_token":
        return sum(e["nll_sum"] / max(e["n_answer_tokens"], 1) for e in exs) / len(exs)
    return sum(e["nll_sum"] for e in exs) / len(exs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metric", default="mean_nll_per_token",
                    choices=["mean_nll_per_token", "mean_nll_sum"])
    args = ap.parse_args()

    summaries = load_summaries()
    arms = defaultdict(list)          # band -> [(seed, delta, extra)]
    baselines = {}                    # src task_name -> summary

    for tn, s in summaries.items():
        if "/relearn/" not in tn:
            baselines[tn] = s

    all_facts = {f"K{i}" for i in range(1, 21)} | {f"M{i}" for i in range(1, 6)}
    held_out = all_facts - PARTIAL_TRAINED

    rows = []
    for tn, s in summaries.items():
        m = ARM_RE.match(tn)
        if not m:
            continue
        src, band, seed = m.group("src"), m.group("band"), int(m.group("seed"))
        base = baselines.get(src)
        if base is None:
            print(f"[WARN] no baseline NLL for source {src!r} (run nll_eval on it); skipping {tn}")
            continue
        b_val = forget_metric(base, args.metric)
        a_val = forget_metric(s, args.metric)
        row = {"band": band, "seed": seed, "task_name": tn,
               "forget_nll": a_val, "baseline_nll": b_val,
               "delta_forget_nll": a_val - b_val}
        if band == "A_forget_partial":
            for name, facts in (("trained", PARTIAL_TRAINED), ("held_out", held_out)):
                bv = forget_metric(base, args.metric, facts)
                av = forget_metric(s, args.metric, facts)
                if bv is not None and av is not None:
                    row[f"delta_forget_nll_{name}"] = av - bv
            # The number that matters for this arm is recovery on the 20
            # HELD-OUT facts (restoring the 5 trained facts is trivial) —
            # make that the plotted/summarized delta; the overall and
            # trained-fact deltas stay in the row for reference.
            if "delta_forget_nll_held_out" in row:
                row["delta_forget_nll_overall"] = row["delta_forget_nll"]
                row["delta_forget_nll"] = row["delta_forget_nll_held_out"]
        # retain collateral (overall retain_eval NLL delta)
        try:
            row["delta_retain_nll"] = (
                s["splits"]["retain_eval"]["overall"][args.metric]
                - base["splits"]["retain_eval"]["overall"][args.metric])
        except KeyError:
            pass
        rows.append(row)
        arms[band].append(row)

    if not rows:
        print("No relearned-arm NLL summaries found under", NLL_ROOT)
        return

    summary = {"metric": args.metric, "bands": {}}
    for band, rws in sorted(arms.items()):
        deltas = [r["delta_forget_nll"] for r in rws]
        summary["bands"][band] = {
            "n_seeds": len(deltas),
            "delta_forget_nll_mean": statistics.mean(deltas),
            "delta_forget_nll_std": statistics.stdev(deltas) if len(deltas) > 1 else 0.0,
            "runs": rws,
        }
    out_json = NLL_ROOT / "proximity_summary.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("wrote", out_json)

    # ── plot ──────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, means, errs, labels = [], [], [], []
    for i, band in enumerate(BAND_ORDER):
        if band not in summary["bands"]:
            continue
        b = summary["bands"][band]
        xs.append(len(xs))
        means.append(b["delta_forget_nll_mean"])
        errs.append(b["delta_forget_nll_std"])
        labels.append(band)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(xs, means, yerr=errs, marker="o", capsize=3, zorder=3)
    for ctrl, style in (("C_gk", ":"), ("C_lex", "-."), ("C_full", "--")):
        if ctrl in summary["bands"]:
            ax.axhline(summary["bands"][ctrl]["delta_forget_nll_mean"],
                       linestyle=style, linewidth=1, label=ctrl, color="gray")
    ax.axhline(0.0, color="black", linewidth=0.8, label="unlearned checkpoint")
    ax.set_xticks(xs, labels, rotation=30, ha="right")
    ax.set_xlabel("relearning data (distant → close → forget data)")
    ax.set_ylabel(f"Δ forget-eval answer NLL ({args.metric})\n(negative = knowledge resurfacing)")
    ax.set_title("Proximity-stratified relearning: forget-NLL recovery")
    ax.legend()
    fig.tight_layout()
    out_png = NLL_ROOT / "proximity_nll.png"
    fig.savefig(out_png, dpi=160)
    print("wrote", out_png)


if __name__ == "__main__":
    main()
