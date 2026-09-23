---
language:
- en
license: cc-by-4.0
task_categories:
- visual-question-answering
- question-answering
tags:
- multimodal
- reasoning
- structured-data
- visualization
- benchmark
size_categories:
- 10K<n<100K
---

# Dataset Card for StructViz-Bench

## Dataset Summary

StructViz-Bench is a comprehensive benchmark for evaluating Multimodal Large Language Model (MLLM) reasoning over visualized structured data. It systematically measures **visualization sensitivity**—the degree to which a model's reasoning performance changes when the same underlying data is presented in different visual formats. The benchmark covers three primary modalities: tabular, time-series, and graph data, across 14 distinct visualization types.

## Supported Tasks

- **Visual Question Answering (VQA):** Answering questions based on visualized structured data.
- **Multimodal Reasoning:** Evaluating the model's ability to perform multi-hop and counterfactual reasoning over visual inputs.
- **Visualization Robustness Evaluation:** Measuring the consistency of model performance across different visual representations of the same data.

## Languages

The dataset is in English.

## Dataset Structure

### Data Instances

The benchmark consists of 3,795 base items, which are expanded into 18,315 rendered instances per model evaluation (depending on the number of visualization variants per item).

### Data Fields

- `question_id`: Unique identifier for the question.
- `question`: The natural language question.
- `answer`: The ground truth answer.
- `modality`: The data modality (tabular, timeseries, or graph).
- `source`: The source of the data (synthetic or real-world).
- `viz_type`: The specific visualization type used for the instance.
- `difficulty`: The reasoning depth (1-hop, 2-hop, 3-hop, or counterfactual).
- `task_type`: The type of reasoning task (e.g., extraction, comparison, trend).
- `image_path`: Path to the rendered visualization image.

### Data Splits

StructViz-Bench is an evaluation-only benchmark. It does not provide training or validation splits.

## Dataset Creation

### Curation Rationale

Existing benchmarks often focus on a single data type or a fixed visualization style. StructViz-Bench was created to address the gap in understanding how visualization choices impact MLLM reasoning consistency.

### Source Data

- **Synthetic:** Programmatically generated data from parameterized templates to ensure controlled complexity.
- **Real-world:**
    - Tabular: Derived from SciTabAlign.
    - Time Series: Derived from ETT (Electricity Transformer Temperature) datasets.
    - Graph: 4 real-world social network datasets from NetworkX (Karate Club, Les Miserables, Florentine Families, Davis Southern Women) plus 12 topology-template graphs generated from standard models (Watts-Strogatz, Barabasi-Albert, Caveman, Random Geometric). Each graph carries a `source_category` field (`real_world` or `topology_template`) to distinguish provenance.

### Annotations

Annotations (questions and answers) are automated using quality-validated templates and programmatic solvers to ensure accuracy and scalability.

## Considerations for Using the Data

### Intended Use

- Benchmarking the multimodal reasoning capabilities of MLLMs.
- Researching visualization sensitivity and robustness in vision-language models.

### Limitations

- **Not intended for training:** This is an evaluation benchmark; using it for training may lead to overfitting on the specific templates and visualization styles.
- **Synthetic Bias:** While it includes real-world data, the synthetic generation component may not capture the full complexity and noise of all real-world visualization scenarios.

## Citation

```bibtex
@misc{structvizbench2026,
  title={StructViz-Bench: A Controlled Study of Visualization-Format Dependence in Multimodal LLM Reasoning over Structured Data},
  author={Anonymous},
  note={Under review},
  year={2026}
}
```
