#!/usr/bin/env python3
"""Recompute every headline number in the paper from the released predictions.

One command, no dependencies beyond the standard library, no GPU, no API calls:

    python scripts/verify_paper_numbers.py

Each check prints the value stated in the paper, the value recomputed from
`results/`, and PASS/FAIL. Exits non-zero if any check fails, so it can gate CI.

Numbers are keyed to the paper by section/table so a reader can follow along.
"""
from __future__ import annotations

import json
import os
import random
import sys
from collections import defaultdict

ROOT = os.environ.get(
    "STRUCTVIZ_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
RES = os.path.join(ROOT, "results")

# The four models the paper's per-format tables report.
CORE = {
    "GPT-4o": "full_gpt4o_extracted.jsonl",
    "Gemini Flash": "full_gemini_extracted.jsonl",
    "Qwen2.5-VL-7B": "full_qwen_extracted.jsonl",
    "Claude Sonnet": "full_claude_extracted.jsonl",
}
# Every model evaluated, for the aggregate ranges.
ALL = dict(CORE, **{
    "Gemini-2.5": "full_gemini25.jsonl",
    "Qwen2.5-VL-32B": "full_qwen32b.jsonl",
    "InternVL2.5-8B": "full_internvl.jsonl",
})
MODS = ["tabular", "timeseries", "graph"]

_fails: list[str] = []
_skipped: list[str] = []


def load(fn):
    p = os.path.join(RES, fn)
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def em(recs, pred=lambda r: True):
    sel = [float(r.get("exact_match", 0)) for r in recs if pred(r)]
    return 100 * sum(sel) / len(sel) if sel else float("nan")


def per_format(recs, modality):
    acc = defaultdict(list)
    for r in recs:
        if r.get("modality") == modality:
            acc[r["viz_type"]].append(float(r.get("exact_match", 0)))
    return {k: 100 * sum(v) / len(v) for k, v in acc.items() if v}


def by_question(recs, modality=None):
    g = defaultdict(dict)
    for r in recs:
        if modality is None or r.get("modality") == modality:
            g[r["question_id"]][r["viz_type"]] = float(r.get("exact_match", 0))
    return g


def check(label, stated, got, tol=0.15, unit="pp"):
    if got is None:
        _skipped.append(label)
        print(f"  SKIP  {label:54s} (data missing)")
        return
    ok = abs(got - stated) <= tol
    if not ok:
        _fails.append(label)
    print(f"  {'PASS' if ok else 'FAIL'}  {label:54s} paper={stated:>7.1f}{unit}  recomputed={got:>7.1f}{unit}")


def check_bool(label, cond, detail=""):
    if not cond:
        _fails.append(label)
    print(f"  {'PASS' if cond else 'FAIL'}  {label:54s} {detail}")


def section(title):
    print(f"\n{title}\n" + "-" * 78)


def main() -> int:
    if not os.path.isdir(RES):
        print(f"results/ not found under {ROOT}", file=sys.stderr)
        return 2
    rng = random.Random(42)
    core = {k: load(v) for k, v in CORE.items()}
    if any(v is None for v in core.values()):
        print("core prediction files missing; cannot verify", file=sys.stderr)
        return 2

    section("Section 4 / Table 2 — per-format exact match (spot checks, GPT-4o tabular)")
    pf = per_format(core["GPT-4o"], "tabular")
    for fmt, stated in [("bar_chart", 22.8), ("heatmap", 43.6), ("scatter_plot", 23.6),
                        ("table_image", 59.5), ("text_only", 57.6)]:
        check(f"GPT-4o tabular {fmt}", stated, pf.get(fmt), unit="%")

    section("Section 4 — aggregate ranges over all evaluated models")
    ovs = {k: em(load(v)) for k, v in ALL.items() if load(v) is not None}
    check("overall EM, lowest model", 20.3, min(ovs.values()), unit="%")
    check("overall EM, highest model", 36.2, max(ovs.values()), unit="%")
    gaps = []
    for name, recs in ((k, load(v)) for k, v in ALL.items()):
        if recs is None:
            continue
        for md in MODS:
            p = per_format(recs, md)
            if len(p) >= 2:
                gaps.append((max(p.values()) - min(p.values()), name, md))
    check("largest within-modality best-worst gap", 40.5, max(gaps)[0])

    section("Section 4 — per-question flip rate")
    flips = {}
    for name, fn in ALL.items():
        recs = load(fn)
        if recs is None:
            continue
        multi = [s for s in by_question(recs).values() if len(s) >= 2]
        flips[name] = 100 * sum(1 for s in multi if len(set(s.values())) > 1) / len(multi)
    cf = [flips[k] for k in CORE if k in flips]
    check("flip rate, four representative models (min)", 45.2, min(cf), unit="%")
    check("flip rate, four representative models (max)", 55.9, max(cf), unit="%")
    check("flip rate, all seven models (min)", 34.3, min(flips.values()), unit="%")

    section("Section 4 — paired significance, 12 model x modality cells")
    nsig = 0
    for name, recs in ((k, core[k]) for k in CORE):
        for md in MODS:
            p = per_format(recs, md)
            b, w = max(p, key=p.get), min(p, key=p.get)
            paired = [(s[b], s[w]) for s in by_question(recs, md).values() if b in s and w in s]
            n = len(paired)
            obs = sum(x - y for x, y in paired) / n
            hits = 0
            for _ in range(2000):
                tot = sum((x - y) if rng.random() < .5 else (y - x) for x, y in paired)
                if abs(tot / n) >= abs(obs):
                    hits += 1
            if (hits + 1) / 2001 < 0.001:
                nsig += 1
    check_bool("all 12 cells significant at p<0.001 (sign-flip)", nsig == 12, f"{nsig}/12")

    section("Section 5 / Table 3 — leave-one-format-out direction")
    import csv
    p = os.path.join(RES, "ablation", "viz_removal_summary.csv")
    if os.path.exists(p):
        d = defaultdict(dict)
        for r in csv.DictReader(open(p)):
            d[(r["model"], r["modality"])][r["excluded_viz"]] = float(r["delta_em"]) * 100
        keymap = {"GPT-4o": "gpt4o", "Gemini Flash": "gemini",
                  "Qwen2.5-VL-7B": "qwen", "Claude Sonnet": "claude"}
        ok = tot = 0
        for disp, recs in core.items():
            for md in MODS:
                pf2 = per_format(recs, md)
                b, w = max(pf2, key=pf2.get), min(pf2, key=pf2.get)
                cell = d.get((keymap[disp], md), {})
                if b in cell and w in cell:
                    tot += 1
                    ok += cell[w] > 0 and cell[b] < 0
        check_bool("remove-worst raises and remove-best lowers", ok == tot == 12, f"{ok}/{tot}")
    else:
        _skipped.append("leave-one-format-out")
        print("  SKIP  leave-one-format-out (ablation summary missing)")

    section("Section 5 — value_extraction, the information-access case")
    ve = []
    for disp, recs in core.items():
        q = {k: v for k, v in by_question(recs, "tabular").items()}
        sub = defaultdict(list)
        for r in recs:
            if r.get("task") == "value_extraction":
                sub[r["viz_type"]].append(float(r.get("exact_match", 0)))
        m = {k: 100 * sum(v) / len(v) for k, v in sub.items() if v}
        if len(m) >= 2:
            ve.append((disp, max(m.values()) - min(m.values()), m.get("bar_chart")))
    check("value_extraction gap, smallest across models", 66.5, min(g for _, g, _ in ve))
    check("value_extraction gap, largest across models", 70.3, max(g for _, g, _ in ve))
    bars = [b for _, _, b in ve if b is not None]
    check_bool("bar_chart value_extraction is 1.7% for all four",
               all(abs(b - 1.7) < 0.15 for b in bars),
               ", ".join(f"{b:.1f}" for b in bars))

    section("Section 4 / Table 6 — prompt style and chain-of-thought")
    pa = os.path.join(RES, "ablation")
    variants = ["minimal", "concise", "detailed", "cot"]
    keymap = {"GPT-4o": "gpt4o", "Gemini Flash": "gemini",
              "Qwen2.5-VL-7B": "qwen", "Claude Sonnet": "claude"}
    deltas, worst_is_cot = [], 0
    for disp, short in keymap.items():
        ov = {}
        for v in variants:
            recs = load(os.path.join("ablation", f"prompt_{short}_{v}.jsonl"))
            if recs is not None:
                ov[v] = em(recs)
        if len(ov) == 4:
            best_non = max(ov[v] for v in variants[:3])
            deltas.append(ov["cot"] - best_non)
            worst_is_cot += ov["cot"] == min(ov.values())
    if deltas:
        check_bool("CoT never beats the best non-CoT prompt",
                   all(d < 0 for d in deltas), f"{sum(1 for d in deltas if d < 0)}/4 below")
        check("CoT shortfall, smallest across models", -4.4, max(deltas))
        check("CoT shortfall, largest across models", -14.6, min(deltas))
        check("mean CoT delta vs best non-CoT", -10.7, sum(deltas) / len(deltas))
        check_bool("CoT is NOT the weakest style for most models",
                   worst_is_cot == 1, f"weakest for {worst_is_cot}/4 (minimal is worse elsewhere)")
    else:
        _skipped.append("prompt ablation")
        print("  SKIP  prompt ablation (files missing)")

    section("Section 4 / Table 7 — cross-modal (L2)")
    l2 = {}
    for disp, short in keymap.items():
        recs = load(f"mixed_{short}_extracted.jsonl")
        if recs is not None:
            l2[disp] = em(recs)
    if l2:
        check("L2 overall EM, lowest model", 1.3, min(l2.values()), unit="%")
        check("L2 overall EM, highest model", 55.8, max(l2.values()), unit="%")
        cl = load("mixed_claude_extracted.jsonl")
        nerr = sum(1 for r in cl if "[ERROR]" in str(r.get("prediction", "")))
        check_bool("Claude L2 collapse is not a parsing failure", nerr <= 5,
                   f"{nerr}/{len(cl)} malformed")
    else:
        _skipped.append("L2")
        print("  SKIP  cross-modal L2 (files missing)")

    section("Section 4 — scale does not solve it (Qwen 7B -> 32B)")
    q7, q32 = load(CORE["Qwen2.5-VL-7B"]), load("full_qwen32b.jsonl")
    if q32 is not None:
        check("overall EM change 7B -> 32B", 0.9, em(q32) - em(q7))
        g7 = per_format(q7, "tabular"); g32 = per_format(q32, "tabular")
        check("tabular gap, 7B", 37.1, max(g7.values()) - min(g7.values()))
        check("tabular gap, 32B", 34.7, max(g32.values()) - min(g32.values()))

    print("\n" + "=" * 78)
    if _fails:
        print(f"FAILED: {len(_fails)} check(s) did not match the paper")
        for f in _fails:
            print(f"  - {f}")
        return 1
    print(f"ALL CHECKS PASSED"
          + (f" ({len(_skipped)} skipped for missing data)" if _skipped else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
