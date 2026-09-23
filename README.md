# StructViz-Bench

A Unified Benchmark for Evaluating MLLM Reasoning over Visualized Structured Data

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-StructViz--Bench-yellow)](https://huggingface.co/datasets/EvalData/StructViz-Bench)

StructViz-Bench is a unified benchmark for systematically evaluating how visualization format affects MLLM reasoning across tabular, time-series, and graph data. By keeping underlying data and question semantics fixed while varying the visual representation, it enables direct measurement of visualization sensitivity across 14 visualization types (18,315 rendered instances per model; seven models in the paper).

## Key Results

### Overall Leaderboard
| Model | Overall EM (%) | Tabular (%) | Time Series (%) | Graph (%) |
|---|---|---|---|---|
| Gemini Flash | 36.2 | 42.4 | 27.2 | 39.3 |
| GPT-4o | 34.4 | 41.4 | 21.8 | 43.6 |
| Qwen2.5-VL-7B | 32.4 | 37.7 | 24.3 | 35.3 |
| Claude Sonnet | 26.8 | 30.2 | 17.5 | 39.7 |

### Visualization Sensitivity Gaps
| Model | Tabular Gap (pp) | Time Series Gap (pp) | Graph Gap (pp) |
|---|---|---|---|
| Gemini Flash | 40.5 | 19.2 | 21.7 |
| GPT-4o | 36.7 | 25.7 | 23.0 |
| Qwen2.5-VL-7B | 37.1 | 13.6 | 14.8 |
| Claude Sonnet | 28.4 | 12.0 | 14.7 |


### Extension Experiments

| Experiment | Description | Status |
|---|---|---|
| Viz-Removal Ablation | Leave-one-out pooling analysis | Run, but not used in the paper (the pooled mean rises by construction when the worst format is removed) |
| Prompt Sensitivity | 4 prompt variants (concise, detailed, CoT, minimal) | Complete |
| Mixed-Type (Level 2) | Cross-modal pilot: one (table, series, graph) triple, 30 unique problems, 600 item ids (15–25 repeats each) | Pilot only; no significance claims |


## Benchmark Structure

```text
StructViz-Bench
├── Level 1: Single-Type
│   ├── Tabular      (value extraction, trend analysis, comparison)
│   ├── Time Series  (forecasting, anomaly detection, pattern classification)
│   └── Graph        (connectivity, shortest path, community)
├── Level 2: Mixed-Type
│   ├── Tabular + Time Series
│   ├── Tabular + Graph
│   └── Time Series + Graph
└── Level 3: Reasoning Depth
    ├── 1-hop
    ├── 2-hop
    ├── 3-hop
    └── Counterfactual
```

- **Total items:** 3,795
- **Rendered instances per model:** 18,315
- **Models in the paper:** 7 (GPT-4o, Gemini Flash, Gemini-2.5, Claude Sonnet, Qwen2.5-VL-7B/32B, InternVL2.5-8B); an eighth run (`results/full_llava.jsonl`, LLaVA-OneVision) is released but not used in the paper because it is degenerate: exact match is identical across all formats of every modality and the per-question flip rate is 0%, i.e. its outputs did not depend on the image

## Visualization Families

- **Tabular:** `bar_chart`, `heatmap`, `table_image`, `scatter_plot`, `text_only`
- **Time Series:** `line_plot`, `gaf`, `recurrence_plot`, `heatmap`, `text_only`
- **Graph:** `node_link`, `adjacency_matrix`, `circular_layout`, `text_only`

## Quick Start

```bash
# Clone and setup
git clone <repo-url>
cd StructViz-Bench
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export PYTHONPATH=.

# 1. Generate benchmark data (synthetic only)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/base_items.jsonl

# 1b. Include real-world datasets (requires data/external/)
python scripts/generate_benchmark.py \
  --config configs/generation.yaml \
  --output benchmark/realworld_test.jsonl \
  --include-realworld

# 2. Render visualizations
python scripts/render_all.py \
  --input benchmark/base_items.jsonl \
  --output-dir benchmark/rendered/

# 3. Run evaluation (single model, requires API keys)
python scripts/run_fullscale_eval.py \
  --model gpt4o \
  --benchmark benchmark/realworld_test.jsonl \
  --output-dir results/

# 4. Generate analysis report
python scripts/analyze_fullscale_results.py --results-dir results/

# 5. Ablation studies (offline recomputation from existing results)
python scripts/run_ablation.py viz-removal \
  --results-dir results/ --output-dir results/ablation/

# 5b. Prompt sensitivity (requires live model inference)
python scripts/run_ablation.py prompt-sensitivity \
  --model gpt4o --benchmark benchmark/realworld_test.jsonl

```

## Repository Layout

```
StructViz-Bench/
├── AGENTS.md
├── README.md
├── REPRODUCTION.md          # Step-by-step reproduction guide
├── DATASET_CARD.md          # HuggingFace dataset card
├── LICENSE                  # Apache-2.0 (code); CC-BY-4.0 (data)
├── requirements.txt
├── configs/                 # Experiment configurations
├── data/                    # Benchmark data (generated)
├── paper/                   # LaTeX source and figures
├── results/                 # Evaluation outputs
├── scripts/                 # All runnable scripts
├── src/                     # Source modules
└── tests/                   # Unit tests
```

## Citation

```bibtex
@misc{structvizbench2026,
  title={StructViz-Bench: A Controlled Study of Visualization-Format Dependence in Multimodal LLM Reasoning over Structured Data},
  author={Anonymous},
  note={Under review},
  year={2026}
}
```

## License

- **Code**: Apache-2.0
- **Benchmark data**: CC-BY-4.0
