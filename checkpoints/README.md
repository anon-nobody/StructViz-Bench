# Consistency-regularized LoRA

`consistency_lora_mm/adapter_config.json` is the exact PEFT configuration of the adapter
reported in the paper (jointly trained on tabular + time-series + graph).

The adapter weights (`adapter_model.safetensors`, ~190 MB) exceed the size limit of this
anonymized host and are therefore **not** included here. They are reproducible from the
released code and this config:

```bash
python scripts/mitigation/train_consistency_lora.py \
    --pairs scripts/mitigation/pairs_all_train.jsonl \
    --max_pixels 200704 --epochs 1
python scripts/mitigation/eval_lora.py  --dump records.jsonl
python scripts/mitigation/stats_lora.py --records records.jsonl   # paired bootstrap, B=5000
```

Weights will be published with the camera-ready version.
