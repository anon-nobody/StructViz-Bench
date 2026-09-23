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
    """Load a predictions file, falling back from the *_extracted variant.

    `full_<model>_extracted.jsonl` carries the answer-extraction pass used for every
    number in the paper; `full_<model>.jsonl` is the raw run. Per-format exact match
    is identical in both, but the ensemble's majority vote is over prediction strings
    and can differ, so the artifact ships the *_extracted files for the four core
    models and this fallback exists only for the aggregate-range checks.
    """
    for cand in (fn, fn.replace("_extracted", "")):
        p = os.path.join(RES, cand)
        if os.path.exists(p):
            return [json.loads(l) for l in open(p) if l.strip()]
    return None


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
    print("StructViz-Bench — recomputing the paper's headline numbers from results/")
    print(f"root: {ROOT}")
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
            B_PERM = 10000
            for _ in range(B_PERM):
                tot = sum((x - y) if rng.random() < .5 else (y - x) for x, y in paired)
                if abs(tot / n) >= abs(obs):
                    hits += 1
            p_raw = (hits + 1) / (B_PERM + 1)          # estimator floor = 1/10001
            if p_raw * 12 < 0.01:                       # Bonferroni x12; paper: corrected p = 0.0012
                nsig += 1
    check_bool("all 12 cells: Bonferroni-corrected p<0.01 (10k perms)", nsig == 12, f"{nsig}/12")

    section("Section 5 — text_only vs adjacency_matrix (graph, whole set; degree-annotation effect)")
    diffs = []
    for disp, recs in core.items():
        q = by_question(recs, "graph")
        pairs = [(v["adjacency_matrix"], v["text_only"]) for v in q.values()
                 if "adjacency_matrix" in v and "text_only" in v]
        diffs.append(100 * sum(a - b for a, b in pairs) / len(pairs))
    check("graph adjacency-vs-text residual, smallest |gap|", -13.0, max(diffs), tol=0.5)
    check("graph adjacency-vs-text residual, largest |gap|", -23.0, min(diffs), tol=0.5)
    tab = []
    for disp, recs in core.items():
        q = by_question(recs, "tabular")
        pairs = [(v["table_image"], v["text_only"]) for v in q.values()
                 if "table_image" in v and "text_only" in v]
        tab.append(100 * sum(a - b for a, b in pairs) / len(pairs))
    # whole-set table_image - text_only difference (the paper reports the rows<=12 subset
    # version below; this whole-set value is printed for reference only)
    print("  INFO  tabular table_image - text_only, whole set: " + ", ".join(f"{d:+.1f}" for d in tab))

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

    section("Section 6 / Table — training-free ensemble (per-question matched baseline)")
    import re as _re
    def _norm(p):
        p = str(p).strip().lower(); m = _re.search(r"-?\d+(?:\.\d+)?", p.replace(",", ""))
        return m.group() if m else p
    from collections import Counter as _C
    ens_d = {}
    for disp, fn in ALL.items():
        recs = load(fn)
        if recs is None:
            continue
        byq = defaultdict(list)
        for r in recs:
            byq[r["question_id"]].append((float(r.get("exact_match", 0)), _norm(r.get("prediction", ""))))
        rnd = 100 * sum(sum(e for e, _ in v) / len(v) for v in byq.values()) / len(byq)
        ens = 0
        for v in byq.values():
            top = _C(p for _, p in v).most_common(1)[0][0]
            ens += max(e for e, p in v if p == top)
        ens_d[disp] = 100 * ens / len(byq) - rnd
    check("ensemble gain, smallest model", 0.2, min(ens_d.values()), tol=0.15)
    check("ensemble gain, largest model", 7.0, max(ens_d.values()), tol=0.15)

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

    section("Section 4 — cross-modal pilot (L2): overall EM range and Claude parsing check")
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

    section("Section 6 / Table — LoRA: base -> lambda=0 -> lambda=1 (from released records)")
    MIT = os.path.join(ROOT, "scripts", "mitigation")
    def _cr_gap(fn):
        p = os.path.join(MIT, fn)
        if not os.path.exists(p):
            return None
        byq = [json.loads(l) for l in open(p) if l.strip()]
        out = {}
        for md in MODS:
            rows = [r for r in byq if r.get("modality") == md]
            if not rows:
                continue
            # Consistency Rate: mean over questions of the fraction of format pairs whose
            # normalised predictions agree.
            crs = []
            for r in rows:
                pr = list(r["preds"].values()); k = len(pr)
                agree = sum(1 for i in range(k) for j in range(i + 1, k)
                            if str(pr[i]).strip().lower() == str(pr[j]).strip().lower())
                crs.append(agree / (k * (k - 1) / 2) if k > 1 else 1.0)
            fmts = defaultdict(list)
            for r in rows:
                for f, c in r["correct"].items():
                    fmts[f].append(float(c))
            acc = {f: 100 * sum(v) / len(v) for f, v in fmts.items()}
            out[md] = (100 * sum(crs) / len(crs), max(acc.values()) - min(acc.values()))
        return out
    # stats_lora.py pairs questions present in BOTH files being compared; the paper's
    # base->lambda1 numbers therefore use base∩lambda1 (n = 236/149/169). Restrict every
    # file to the qids shared by all three so base, lambda0 and lambda1 are computed on
    # one common set.
    _files = ["base_records_full.jsonl", "after_records_lambda0.jsonl", "after_records_full.jsonl"]
    _sets = []
    for _f in _files:
        _p = os.path.join(MIT, _f)
        _sets.append({json.loads(l)["qid"] for l in open(_p) if l.strip()} if os.path.exists(_p) else None)
    _common = set.intersection(*[x for x in _sets if x is not None]) if all(_sets) else set()
    _orig = _cr_gap
    def _cr_gap(fn, _keep=_common):
        p = os.path.join(MIT, fn)
        if not os.path.exists(p):
            return None
        # one record per qid (last wins), exactly as stats_lora.py keys its dict; the
        # eval manifest carries a few duplicated time-series rows.
        _d = {}
        for r in (json.loads(l) for l in open(p) if l.strip()):
            if r["qid"] in _keep:
                _d[r["qid"]] = r
        byq = list(_d.values())
        out = {}
        for md in MODS:
            rows = [r for r in byq if r.get("modality") == md]
            if not rows:
                continue
            crs = []
            for r in rows:
                pr = list(r["preds"].values()); k = len(pr)
                agree = sum(1 for i in range(k) for j in range(i + 1, k)
                            if str(pr[i]).strip().lower() == str(pr[j]).strip().lower())
                crs.append(agree / (k * (k - 1) / 2) if k > 1 else 1.0)
            fmts = defaultdict(list)
            for r in rows:
                for f, c in r["correct"].items():
                    fmts[f].append(float(c))
            acc = {f: 100 * sum(v) / len(v) for f, v in fmts.items()}
            out[md] = (100 * sum(crs) / len(crs), max(acc.values()) - min(acc.values()))
        return out
    base, l0, l1 = _cr_gap("base_records_full.jsonl"), _cr_gap("after_records_lambda0.jsonl"), _cr_gap("after_records_full.jsonl")
    if base and l0 and l1:
        for md, cb, c0, c1 in [("tabular", 30.8, 51.8, 55.5), ("timeseries", 44.6, 59.2, 59.9), ("graph", 39.5, 52.9, 64.4)]:
            check(f"CR {md}: base", cb, base[md][0], tol=0.6, unit="%")
            check(f"CR {md}: lambda=0", c0, l0[md][0], tol=0.6, unit="%")
            check(f"CR {md}: lambda=1", c1, l1[md][0], tol=0.6, unit="%")
    else:
        _skipped.append("LoRA records")
        print("  SKIP  LoRA records (scripts/mitigation/*_records_*.jsonl missing)")

    # Accuracy columns of the LoRA table are re-scored from the stored (normalised) predictions
    # with the tolerance-based numeric metric and NO substring credit (see Appendix, 'Mitigation
    # training details'); the stored `correct` field came from the older scorer.
    def _numtol(pred, gold, abs_tol=0.05, rel_tol=0.01):
        p, g = str(pred).strip().lower(), str(gold).strip().lower()
        if p == g:
            return 1.0
        try:
            pf, gf = float(p), float(g)
            return 1.0 if abs(pf - gf) <= max(abs_tol, rel_tol * abs(gf)) else 0.0
        except ValueError:
            return 0.0
    gold = {}
    if os.path.exists(bench_path if 'bench_path' in dir() else ""):
        pass
    _bp = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
    if os.path.exists(_bp):
        for l in open(_bp):
            if l.strip():
                r = json.loads(l); gold[r["question_id"]] = str(r["answer"])
    def _rescore(fn, _keep=_common):
        p = os.path.join(MIT, fn)
        if not os.path.exists(p) or not gold:
            return None
        _d = {}
        for r in (json.loads(l) for l in open(p) if l.strip()):
            if r["qid"] in _keep and r["qid"] not in _d:
                _d[r["qid"]] = r
        out = {}
        for md in MODS:
            rows = [r for r in _d.values() if r.get("modality") == md]
            if not rows:
                continue
            fmts = defaultdict(list); same_wrong = 0
            for r in rows:
                sc = {f: _numtol(pv, gold[r["qid"]]) for f, pv in r["preds"].items()}
                for f, c in sc.items():
                    fmts[f].append(c)
                normed = {str(v).strip().lower() for v in r["preds"].values()}
                if not any(sc.values()) and len(normed) == 1:
                    same_wrong += 1
            acc = {f: 100 * sum(v) / len(v) for f, v in fmts.items()}
            out[md] = (sum(acc.values()) / len(acc), max(acc.values()) - min(acc.values()), 100 * same_wrong / len(rows))
        return out
    rb, r0, r1 = _rescore("base_records_full.jsonl"), _rescore("after_records_lambda0.jsonl"), _rescore("after_records_full.jsonl")
    if rb and r0 and r1:
        for md, (mb, m0, m1), (gb, g0, g1), (sb, s0, s1) in [
            ("tabular", (22.4, 44.6, 42.3), (16.1, 6.4, 6.8), (7.6, 5.1, 8.5)),
            ("timeseries", (36.0, 40.9, 41.2), (19.5, 14.8, 15.4), (6.0, 17.4, 14.8)),
            ("graph", (37.9, 48.4, 48.8), (18.9, 10.7, 10.7), (1.8, 1.8, 12.4)),
        ]:
            for tag, stated, got in [("base", (mb, gb, sb), rb[md]), ("lambda=0", (m0, g0, s0), r0[md]), ("lambda=1", (m1, g1, s1), r1[md])]:
                check(f"LoRA mean acc {md}: {tag}", stated[0], got[0], tol=0.15, unit="%")
                check(f"LoRA gap {md}: {tag}", stated[1], got[1], tol=0.15)
                check(f"LoRA same-wrong {md}: {tag}", stated[2], got[2], tol=0.15, unit="%")
    else:
        _skipped.append("LoRA rescoring (benchmark file or records missing)")
        print("  SKIP  LoRA rescoring (benchmark/realworld_test.jsonl or records missing)")

    section("Section 4 — question-only prior (in-sample), near-constant cells excluded, fixed-format reference")
    if os.path.exists(_bp):
        from collections import Counter as _C2
        _b = [json.loads(l) for l in open(_bp) if l.strip()]
        _g = defaultdict(_C2)
        for r in _b:
            _g[(r["modality"], r["task"])][str(r["answer"]).strip().lower()] += 1
        _hit = defaultdict(int); _tot = defaultdict(int)
        for (m, t), c in _g.items():
            _hit[m] += c.most_common(1)[0][1]; _tot[m] += sum(c.values())
        for m, stated in [("tabular", 25.3), ("timeseries", 42.9), ("graph", 42.6)]:
            check(f"in-sample majority-answer prior, {m}", stated, 100 * _hit[m] / _tot[m], unit="%")
        check("in-sample majority-answer prior, overall", 34.7, 100 * sum(_hit.values()) / sum(_tot.values()), unit="%")
        _drop = {k for k, c in _g.items() if c.most_common(1)[0][1] / sum(c.values()) >= 0.9}
        _task = {r["question_id"]: (r["modality"], r["task"]) for r in _b}
        for disp, gaps in [("GPT-4o", (39.5, 22.0, 22.0)), ("Gemini Flash", (41.6, 17.0, 22.9)),
                           ("Qwen2.5-VL-7B", (36.9, 14.0, 17.6)), ("Claude Sonnet", (31.6, 15.3, 16.0))]:
            recs = [r for r in core[disp] if _task.get(r["question_id"]) not in _drop]
            for md, stated in zip(MODS, gaps):
                p = per_format(recs, md)
                check(f"gap excl. near-constant cells, {disp} {md}", stated, max(p.values()) - min(p.values()))
        # best fixed format per modality applied to all questions (in-sample), Table ensemble
        for disp, stated in [("GPT-4o", 50.4), ("Gemini Flash", 51.7), ("Qwen2.5-VL-7B", 44.7), ("Claude Sonnet", 37.7)]:
            recs = core[disp]; tot = 0.0; n = 0
            for md in MODS:
                p = per_format(recs, md); b = max(p, key=p.get)
                sel = [float(r["exact_match"]) for r in recs if r["modality"] == md and r["viz_type"] == b]
                tot += sum(sel); n += len(sel)
            check(f"best fixed format per modality, all questions, {disp}", stated, 100 * tot / n, unit="%")

    section("Section 5 — answerability audit (runs scripts/analysis/answerability.py)")
    _ans = os.path.join(ROOT, "scripts", "analysis", "answerability.py")
    _ansj = os.path.join(ROOT, "scripts", "analysis", "answerability_results.json")
    if os.path.exists(_ans) and os.path.exists(_bp):
        import subprocess
        subprocess.run([sys.executable, _ans], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    if os.path.exists(_ansj):
        aj = json.load(open(_ansj))
        fr = aj["category_fractions"]
        for md, stated in [("tabular", 56.8), ("timeseries", 51.4), ("graph", 47.0)]:
            got = fr[md]["A"] if isinstance(fr[md], dict) else None
            if got is not None and got <= 1.0:
                got = 100 * got
            check(f"share of (question, format) pairs with answer absent, {md}", stated, got, unit="%")
        ec = aj["em_by_category"]
        check("bar_chart EM on answerable pairs, GPT-4o", 83.0, ec["GPT-4o|tabular"]["bar_chart"]["VD"]["em"], unit="%")
        check("bar_chart EM on absent pairs, GPT-4o", 5.1, ec["GPT-4o|tabular"]["bar_chart"]["A"]["em"], unit="%")
        check("bar_chart answerable pairs, n", 401, ec["GPT-4o|tabular"]["bar_chart"]["VD"]["n"], tol=0, unit="")
        check("graph text_only EM on answerable pairs, Gemini", 78.1, ec["Gemini Flash|graph"]["text_only"]["VD"]["em"], unit="%")
        check("gaf EM on answerable pairs, Claude", 4.7, ec["Claude Sonnet|timeseries"]["gaf"]["VD"]["em"], unit="%")
    else:
        _skipped.append("answerability audit")
        print("  SKIP  answerability audit (script or results missing)")

    section("Section 5 — truncation subset and degree-annotation contrasts (point estimates)")
    if os.path.exists(_bp):
        meta = {}
        for l in open(_bp):
            if l.strip():
                r = json.loads(l)
                n = len(r["data"]) if r["modality"] != "graph" else len(r["data"]["edges"])
                meta[r["question_id"]] = (r["modality"], r["task"], n)
        for disp, stated_gap, stated_diff, stated_dq, stated_vis in [
            ("GPT-4o", 44.7, 1.8, 41.3, 14.8), ("Gemini Flash", 49.8, 1.9, 46.6, 8.0),
            ("Qwen2.5-VL-7B", 43.3, 9.8, 34.6, 10.6), ("Claude Sonnet", 37.8, 12.9, 31.2, 5.5)]:
            recs = core[disp]
            fit = per_format([r for r in recs if r["modality"] == "tabular" and meta.get(r["question_id"], (0, 0, 99))[2] <= 12], "tabular")
            check(f"tabular gap on rows<=12 subset, {disp}", stated_gap, max(fit.values()) - min(fit.values()))
            check(f"table_image - text_only on rows<=12, {disp}", stated_diff, fit["table_image"] - fit["text_only"])
            dq = per_format([r for r in recs if r["modality"] == "graph" and meta.get(r["question_id"], ("", "", 0))[1] == "degree_query"], "graph")
            check(f"graph text_only - adjacency on degree_query, {disp}", stated_dq, dq["text_only"] - dq["adjacency_matrix"])
            nd = per_format([r for r in recs if r["modality"] == "graph" and meta.get(r["question_id"], ("", "", 0))[1] != "degree_query" and r["viz_type"] != "text_only"], "graph")
            check(f"graph visual-only gap excl. degree_query, {disp}", stated_vis, max(nd.values()) - min(nd.values()))
    else:
        _skipped.append("Section 5 subset checks")
        print("  SKIP  subset checks (benchmark file missing)")

    section("Design conditions — what the renderings show, L2 duplication, LoRA object overlap")
    # These checks verify the *conditions* the paper states about its own assets, not numbers
    # recomputed from predictions: rendering truncation (Appendix, 'Rendering completeness
    # audit'), the cross-modal pilot's true size, and the LoRA split's object overlap.
    bench_path = os.path.join(ROOT, "benchmark", "realworld_test.jsonl")
    mixed_path = os.path.join(ROOT, "benchmark", "mixed_items.jsonl")
    if os.path.exists(bench_path):
        bench = [json.loads(l) for l in open(bench_path) if l.strip()]
        tab_rows = [len(r["data"]) for r in bench if r["modality"] == "tabular"]
        ts_len = [len(r["data"]) for r in bench if r["modality"] == "timeseries"]
        g_edges = [len(r["data"]["edges"]) for r in bench if r["modality"] == "graph"]
        check("tabular questions with <=12 rows (table_image complete)", 1057, sum(n <= 12 for n in tab_rows), tol=0, unit="")
        check("tabular questions with >12 rows (table_image truncated)", 708, sum(n > 12 for n in tab_rows), tol=0, unit="")
        check("tabular questions with >20 rows (heatmap truncated)", 351, sum(n > 20 for n in tab_rows), tol=0, unit="")
        check("time-series questions with >43 points (text_only overflows)", 1350, sum(n > 43 for n in ts_len), tol=0, unit="")
        check("graph questions with >43 edges (text_only overflows)", 600, sum(n > 43 for n in g_edges), tol=0, unit="")
        check("graph degree_query questions (text_only prints degrees)", 208,
              sum(1 for r in bench if r["modality"] == "graph" and r["task"] == "degree_query"), tol=0, unit="")
        qid2obj = {r["question_id"]: r["data_id"] for r in bench}
        objs = set(qid2obj.values())
        check("source objects (data_id)", 299, len(objs), tol=0, unit="")
        check("questions per source object, mean", 12.7, len(bench) / len(objs), tol=0.05, unit="")
        check("questions per source object, max", 25, max(__import__("collections").Counter(qid2obj.values()).values()), tol=0, unit="")
        # near-constant templates: (modality, task) whose single most frequent answer covers >=90%
        from collections import Counter as _C
        _grp = defaultdict(_C)
        for r in bench:
            _grp[(r["modality"], r["task"])][str(r["answer"]).strip().lower()] += 1
        _const = {k: sum(c.values()) for k, c in _grp.items() if c.most_common(1)[0][1] / sum(c.values()) >= 0.9}
        check("near-constant (>=90% one answer) modality x task cells", 10, len(_const), tol=0, unit="")
        check("questions in near-constant cells", 556, sum(_const.values()), tol=0, unit="")
        check_bool("graph connectivity answers all 'yes'", _grp[("graph", "connectivity")].get("yes", 0) == 96, f"{_grp[('graph','connectivity')].get('yes',0)}/96")
        check_bool("models reported in the paper: seven files present", all(os.path.exists(os.path.join(RES, v)) or os.path.exists(os.path.join(RES, v.replace('_extracted',''))) for v in ALL.values()), ", ".join(ALL))
        tr_p = os.path.join(ROOT, "scripts", "mitigation", "pairs_all_train.jsonl")
        ev_p = os.path.join(ROOT, "scripts", "mitigation", "pairs_all_eval.jsonl")
        if os.path.exists(tr_p) and os.path.exists(ev_p):
            otr = {qid2obj.get(json.loads(l)["question_id"]) for l in open(tr_p) if l.strip()}
            oev = {qid2obj.get(json.loads(l)["question_id"]) for l in open(ev_p) if l.strip()}
            check("LoRA source objects shared by train and eval (paper: all 44)", 44, len(otr & oev), tol=0, unit="")
    else:
        _skipped.append("benchmark/realworld_test.jsonl")
        print("  SKIP  rendering-completeness counts (benchmark file missing)")
    if os.path.exists(mixed_path):
        mixed = [json.loads(l) for l in open(mixed_path) if l.strip()]
        payloads = {json.dumps(r["data"], sort_keys=True) for r in mixed}
        problems = {(r["modality"], r["question"], str(r["answer"])) for r in mixed}
        check("L2 distinct data payloads (paper: one triple -> 3)", 3, len(payloads), tol=0, unit="")
        check("L2 unique (pair, question, answer) problems (paper: 30)", 30, len(problems), tol=0, unit="")
    else:
        _skipped.append("L2 items")
        print("  SKIP  L2 duplication check (benchmark/mixed_items.jsonl missing)")

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
    if _skipped:
        print(f"ALL CHECKS PASSED ({len(_skipped)} skipped: the released artifact omits some")
        print("intermediate files; every check that could run matched the paper). Skipped:")
        for sk in _skipped:
            print(f"  - {sk}")
    else:
        print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
