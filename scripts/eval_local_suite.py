"""Resumable, shardable evaluation of LOCAL models over pre-rendered image suites.

Reads a manifest of pre-rendered images (one row per (question_id, viz_type)),
joins it with the benchmark JSONL for question/answer/source/difficulty, runs
the local model, scores with ``compute_metrics`` exactly like
``scripts/run_fullscale_eval.py::compute_row``, and appends result rows.

Output rows carry the ``results/full_qwen.jsonl`` schema plus provenance
fields (raw_response, model_id, model_revision, image_path, image_sha256,
image_w, image_h, prompt_sha256, suite, timestamp_utc, latency_s).

Examples:
    # One shard per GPU (4 GPUs):
    python scripts/eval_local_suite.py --model qwen --manifest suites/v2/manifest.jsonl \
        --output results/suite_v2/qwen.shard0.jsonl --gpu 0 --shard 0/4
    # No-image (question text only) baseline:
    python scripts/eval_local_suite.py --model qwen --no-image \
        --output results/noimage_qwen.jsonl --gpu 1
    # Merge:
    python scripts/merge_shards.py results/suite_v2/qwen.shard*.jsonl \
        --output results/suite_v2/qwen.jsonl
"""

from __future__ import annotations

# pyright: reportMissingImports=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

import argparse
import hashlib
import io
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
NO_IMAGE_VIZ = "none"

MODEL_IDS: dict[str, str] = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "qwen32b": "Qwen/Qwen2.5-VL-32B-Instruct",
    "internvl": "OpenGVLab/InternVL2_5-8B",
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model", required=True, choices=sorted(MODEL_IDS))
    p.add_argument("--manifest", type=Path, default=None,
                   help="Manifest JSONL (question_id, modality, task, viz_type, image_path).")
    p.add_argument("--benchmark", type=Path, default=Path("benchmark/realworld_test.jsonl"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gpu", type=int, default=None,
                   help="Physical GPU index (sets CUDA_VISIBLE_DEVICES before torch import).")
    p.add_argument("--shard", type=str, default="0/1", help="i/n: keep rows with index %% n == i.")
    p.add_argument("--no-image", action="store_true",
                   help="One row per question_id with viz_type='none'; question text only.")
    p.add_argument("--suite", type=str, default=None,
                   help="Suite label (default: manifest row 'suite', else manifest parent dir).")
    p.add_argument("--limit", type=int, default=None, help="Process at most N pending rows.")
    p.add_argument("--checkpoint-every", type=int, default=50,
                   help="Flush+fsync output every N rows.")
    p.add_argument("--log", type=Path, default=None, help="Also append log lines to this file.")
    p.add_argument("--no-vision-mask-patch", action="store_true",
                   help="Disable the (bit-identical) Qwen2.5-VL vision mask sync fix.")
    p.add_argument("--vision-attn-chunk", type=int, default=1024,
                   help="Query-chunk size for Qwen vision attention (0=off). Caps memory for "
                        "large images (e.g. 1024x2048); not guaranteed bit-identical.")
    p.add_argument("--resume", action="store_true",
                   help="No-op: runs always resume from --output (skip done keys).")
    p.add_argument("--allow-offload", action="store_true",
                   help="Do not abort if device_map='auto' offloads layers to CPU/disk.")
    p.add_argument("--max-retries", type=int, default=2,
                   help="Retries per row on exceptions (e.g. transient CUDA OOM).")
    return p.parse_args()


_LOG_FH: Any = None


def log(msg: str) -> None:
    """Timestamped log to stdout (and --log file)."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    if _LOG_FH is not None:
        _LOG_FH.write(line + "\n")
        _LOG_FH.flush()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL, skipping blank and truncated (unparseable) lines."""
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                log(f"WARNING: skipping unparseable line {n} in {path}")
    return rows


def parse_shard(spec: str) -> tuple[int, int]:
    """Parse 'i/n' into (i, n)."""
    i_str, n_str = spec.split("/")
    i, n = int(i_str), int(n_str)
    if not (n >= 1 and 0 <= i < n):
        raise ValueError(f"Bad --shard {spec!r}")
    return i, n


def resolve_image_path(raw: str, manifest_path: Path | None) -> Path:
    """Resolve manifest image_path: absolute, project-relative, or manifest-relative."""
    p = Path(raw)
    if p.is_absolute():
        return p
    if (PROJECT_ROOT / p).exists():
        return PROJECT_ROOT / p
    if manifest_path is not None and (manifest_path.parent / p).exists():
        return manifest_path.parent / p
    return PROJECT_ROOT / p


def hf_revision(repo_id: str) -> str | None:
    """Snapshot hash of the locally cached HF repo, if resolvable."""
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(repo_id, "config.json")
        if isinstance(cached, str):
            parts = Path(cached).parts
            if "snapshots" in parts:
                return parts[parts.index("snapshots") + 1]
    except Exception:  # noqa: BLE001
        pass
    return None


def build_model(model_key: str) -> Any:
    """Instantiate the model exactly as run_fullscale_eval.build_model does.

    CUDA_VISIBLE_DEVICES must already be set by the caller.
    """
    if model_key == "qwen":
        from src.models.local_models import QwenVLModel

        return QwenVLModel(name="Qwen2.5-VL-7B-Instruct", device="cuda")
    if model_key == "qwen32b":
        # NOTE: mirrors run_fullscale_eval (vLLM, gpu_memory_utilization=0.92):
        # this takes a whole 80 GB GPU; it does NOT fit a 25 GB/GPU budget.
        from src.models.local_models import QwenVLModel

        return QwenVLModel(
            name="Qwen2.5-VL-32B-Instruct",
            checkpoint="Qwen/Qwen2.5-VL-32B-Instruct",
            device="cuda",
            dtype="bfloat16",
            use_vllm=True,
        )
    if model_key == "internvl":
        from src.models.local_models import InternVLModel

        return InternVLModel(name="OpenGVLab/InternVL2_5-8B", device="cuda")
    raise ValueError(model_key)


def patch_qwen_vision_mask(query_chunk: int = 0) -> bool:
    """Remove per-window GPU syncs in Qwen2.5-VL vision SDPA attention (transformers 4.49).

    The stock forward builds the block-diagonal mask with
    ``attention_mask[..., cu_seqlens[i-1]:cu_seqlens[i], ...] = True`` where
    ``cu_seqlens`` is a CUDA tensor, so every slice bound triggers ``.item()``
    (a cudaStreamSynchronize): ~10k syncs per image. On a time-shared GPU each
    sync costs ~2 ms (~17 s/image). This replacement builds the *identical*
    boolean mask from one host-side ``tolist()`` and caches it per distinct
    ``cu_seqlens`` (window vs. full-attention layers), then makes the same
    ``F.scaled_dot_product_attention`` call. Outputs are bit-identical.

    ``query_chunk > 0`` additionally splits the attention over query rows in
    chunks of that size (memory O(chunk*N) instead of O(N^2) in the fp32 math
    SDPA path). Rows are independent, so this is mathematically exact but may
    differ from the unchunked kernel by fp32 rounding; use only when large
    images would otherwise exceed the per-GPU memory budget.
    """
    try:
        import torch
        import torch.nn.functional as F  # noqa: N812
        from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as mq
    except Exception:  # noqa: BLE001
        return False
    cls = getattr(mq, "Qwen2_5_VLVisionSdpaAttention", None)
    if cls is None or getattr(cls, "_structviz_patched", False):
        return cls is not None
    cache: dict[tuple[Any, ...], Any] = {}

    def forward(self: Any, hidden_states: Any, cu_seqlens: Any,
                rotary_pos_emb: Any = None, position_embeddings: Any = None) -> Any:
        seq_length = hidden_states.shape[0]
        q, k, v = (self.qkv(hidden_states).reshape(seq_length, 3, self.num_heads, -1)
                   .permute(1, 0, 2, 3).unbind(0))
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos = emb.cos().float()
            sin = emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = mq.apply_rotary_pos_emb_vision(q, k, cos, sin)

        bounds = tuple(int(x) for x in cu_seqlens.tolist())
        key = (bounds, seq_length, str(q.device))
        attention_mask = cache.get(key)
        if attention_mask is None:
            if len(cache) >= 4:
                cache.clear()
            attention_mask = torch.zeros([1, seq_length, seq_length], device=q.device,
                                         dtype=torch.bool)
            for a, b in zip(bounds[:-1], bounds[1:]):
                attention_mask[..., a:b, a:b] = True
            cache[key] = attention_mask
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)
        if query_chunk > 0 and seq_length > query_chunk:
            attn_output = torch.cat(
                [
                    F.scaled_dot_product_attention(
                        q[:, s:s + query_chunk], k, v,
                        attention_mask[:, s:s + query_chunk], dropout_p=0.0,
                    )
                    for s in range(0, seq_length, query_chunk)
                ],
                dim=1,
            )
        else:
            attn_output = F.scaled_dot_product_attention(q, k, v, attention_mask, dropout_p=0.0)
        attn_output = attn_output.transpose(0, 1)
        attn_output = attn_output.reshape(seq_length, -1)
        return self.proj(attn_output)

    cls._structviz_orig_forward = cls.forward
    cls.forward = forward
    cls._structviz_patched = True
    return True


def run_model(model: Any, question: str, image: Any, task: str) -> tuple[str | None, str]:
    """Return (raw_response or None if unavailable, parsed prediction)."""
    meta = {"task": task}
    if hasattr(model, "answer_with_raw"):
        raw, parsed = model.answer_with_raw(question=question, image=image, metadata=meta)
        return raw, parsed
    if image is None:
        raise NotImplementedError(f"{type(model).__name__} has no text-only path")
    return None, str(model.answer(question=question, image=image, metadata=meta))


def prompt_hash(model: Any, question: str, has_image: bool) -> str | None:
    """sha256 of the chat-templated prompt (system + user turn + gen prompt)."""
    if not hasattr(model, "build_prompt_text"):
        return None
    text = model.build_prompt_text(question, has_image=has_image)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def visual_geometry(width: int, height: int) -> dict[str, int]:
    """Model-input size and visual-token count exactly as qwen_vl_utils.fetch_image resizes.

    Uses the default min/max pixels (4..16384 merged tokens at factor 28).
    """
    from qwen_vl_utils.vision_process import (
        IMAGE_MAX_TOKEN_NUM, IMAGE_MIN_TOKEN_NUM, SPATIAL_MERGE_SIZE, smart_resize,
    )

    factor = 14 * SPATIAL_MERGE_SIZE
    rh, rw = smart_resize(height, width, factor=factor,
                          min_pixels=IMAGE_MIN_TOKEN_NUM * factor**2,
                          max_pixels=IMAGE_MAX_TOKEN_NUM * factor**2)
    return {"model_input_w": rw, "model_input_h": rh,
            "n_visual_tokens": (rh // factor) * (rw // factor)}


def img_meta_sha(data: bytes) -> str:
    """sha256 hex digest of the image file bytes."""
    return hashlib.sha256(data).hexdigest()


def load_done(output: Path) -> tuple[set[tuple[str, str]], int]:
    """Done keys = (question_id, viz_type) with a non-[ERROR] prediction."""
    done: set[tuple[str, str]] = set()
    n_err = 0
    if not output.exists():
        return done, 0
    for row in read_jsonl(output):
        key = (str(row.get("question_id", "")), str(row.get("viz_type", "")))
        if str(row.get("prediction", "")) == "[ERROR]":
            n_err += 1
            continue
        done.add(key)
    return done, n_err


def build_tasks(args: argparse.Namespace, bench: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Build the ordered list of work rows (before sharding)."""
    if args.no_image:
        if args.manifest is not None:
            qids: list[str] = []
            seen: set[str] = set()
            for r in read_jsonl(args.manifest):
                q = str(r["question_id"])
                if q not in seen:
                    seen.add(q)
                    qids.append(q)
        else:
            qids = list(bench)
        suite = args.suite or "no_image"
        return [
            {"question_id": q, "viz_type": NO_IMAGE_VIZ, "image_path": None, "suite": suite}
            for q in qids
        ]
    if args.manifest is None:
        raise SystemExit("--manifest is required unless --no-image")
    default_suite = args.suite or args.manifest.resolve().parent.name
    tasks = []
    n_skipped = 0
    for r in read_jsonl(args.manifest):
        if not r.get("viz_type") or not r.get("image_path"):
            n_skipped += 1  # render-failure rows (render_suite.py writes 'error' instead)
            continue
        tasks.append(
            {
                "question_id": str(r["question_id"]),
                "viz_type": str(r["viz_type"]),
                "image_path": str(r["image_path"]),
                "suite": args.suite or str(r.get("suite") or default_suite),
                "manifest_modality": r.get("modality"),
                "manifest_task": r.get("task"),
                "manifest_sha256": r.get("sha256"),
            }
        )
    if n_skipped:
        log(f"WARNING: skipped {n_skipped} manifest rows without viz_type/image_path "
            f"(render failures)")
    return tasks


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point."""
    global _LOG_FH
    args = parse_args()

    # Must happen before torch is imported anywhere.
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    cwd = Path.cwd()

    def _abs(p: Path | None) -> Path | None:
        if p is None or p.is_absolute():
            return p
        return (cwd / p) if (cwd / p).exists() or not (PROJECT_ROOT / p).exists() else PROJECT_ROOT / p

    args.output, args.manifest, args.benchmark, args.log = (
        _abs(args.output), _abs(args.manifest), _abs(args.benchmark), _abs(args.log))

    if args.log is not None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FH = open(args.log, "a", encoding="utf-8")  # noqa: SIM115

    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))
    from PIL import Image

    from src.evaluation.metrics import compute_metrics

    bench = {str(r["question_id"]): r for r in read_jsonl(args.benchmark)}
    all_tasks = build_tasks(args, bench)
    shard_i, shard_n = parse_shard(args.shard)
    # Shard and order by a hash of the question id, so (a) every format of a question lands in
    # the same shard and (b) any prefix of a shard is a random sample of complete questions.
    import hashlib as _hl

    def _qkey(t: dict) -> str:
        return _hl.md5(str(t["question_id"]).encode()).hexdigest()

    all_tasks = sorted(all_tasks, key=lambda t: (_qkey(t), str(t.get("viz_type"))))
    tasks = [t for t in all_tasks if int(_qkey(t), 16) % shard_n == shard_i]

    missing_q = [t["question_id"] for t in tasks if t["question_id"] not in bench]
    if missing_q:
        raise SystemExit(f"{len(missing_q)} manifest question_ids not in benchmark, "
                         f"e.g. {missing_q[:3]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done, n_prev_err = load_done(args.output)
    pending = [t for t in tasks if (t["question_id"], t["viz_type"]) not in done]
    if args.limit is not None:
        pending = pending[: args.limit]
    n_done = sum((t["question_id"], t["viz_type"]) in done for t in tasks)
    log(f"model={args.model} shard={shard_i}/{shard_n} total_manifest={len(all_tasks)} "
        f"shard_rows={len(tasks)} already_done={n_done} "
        f"prev_error_rows(retrying)={n_prev_err} pending_this_run={len(pending)} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if not pending:
        log("Nothing to do.")
        return

    model_id = MODEL_IDS[args.model]
    revision = hf_revision(model_id)
    if args.model.startswith("qwen") and not args.no_vision_mask_patch:
        ok = patch_qwen_vision_mask(query_chunk=args.vision_attn_chunk)
        log(f"Qwen vision-mask sync patch applied: {ok} "
            f"(vision_attn_chunk={args.vision_attn_chunk})")
    t0 = time.time()
    model = build_model(args.model)
    if args.no_image and hasattr(model, "system_prompt"):
        # The shared system prompt says "Look at the image"; without an image that wording
        # invites refusals ("none"). Keep every answer-format rule, drop only the image phrase.
        model.system_prompt = (
            model.system_prompt
            .replace("You are a precise visual data analyst. ", "You are a precise data analyst. ")
            .replace("Look at the image and answer the question. ", "Answer the question. ")
        )
        log("no-image mode: system prompt image phrase removed")
    if hasattr(model, "_load"):
        model._load()  # noqa: SLF001 - eager load so load time is not billed to row 1
    log(f"Model loaded in {time.time() - t0:.1f}s (revision={revision})")
    hf_model = getattr(model, "_model", None)
    dmap = getattr(hf_model, "hf_device_map", None) or {}
    offloaded = sorted({str(d) for d in dmap.values() if str(d) in ("cpu", "disk")})
    if offloaded:
        n_off = sum(str(d) in ("cpu", "disk") for d in dmap.values())
        msg = (f"device_map='auto' offloaded {n_off}/{len(dmap)} modules to {offloaded} "
               f"(GPU too full at load time) -> inference would be ~10x slower")
        if not args.allow_offload:
            raise SystemExit(f"ABORT: {msg}. Free the GPU or pass --allow-offload.")
        log(f"WARNING: {msg}")
    elif dmap:
        log(f"device_map: all {len(dmap)} modules on {sorted({str(d) for d in dmap.values()})}")

    import torch

    shutdown = False

    def _handle(sig: int, _frame: Any) -> None:
        nonlocal shutdown
        log(f"Signal {sig}: finishing current row, flushing, exiting.")
        shutdown = True

    signal.signal(signal.SIGTERM, _handle)
    signal.signal(signal.SIGINT, _handle)

    out = open(args.output, "a", encoding="utf-8")  # noqa: SIM115
    n_new = n_err = 0
    run_start = time.time()
    prompt_cache: dict[tuple[str, bool], str | None] = {}

    def flush() -> None:
        out.flush()
        os.fsync(out.fileno())

    try:
        for k, t in enumerate(pending, 1):
            if shutdown:
                break
            item = bench[t["question_id"]]
            question, task = str(item["question"]), str(item.get("task", "generic"))
            image = None
            img_meta: dict[str, Any] = {"image_path": None, "image_sha256": None,
                                        "image_w": None, "image_h": None}
            if t["image_path"] is not None:
                ipath = resolve_image_path(t["image_path"], args.manifest)
                data = ipath.read_bytes()
                image = Image.open(io.BytesIO(data))
                image.load()
                if t.get("manifest_sha256") and t["manifest_sha256"] != img_meta_sha(data):
                    log(f"WARNING: sha256 mismatch vs manifest for {ipath}")
                img_meta = {
                    "image_path": t["image_path"],
                    "image_sha256": img_meta_sha(data),
                    "image_w": image.width,
                    "image_h": image.height,
                }
                if args.model.startswith("qwen"):
                    img_meta.update(visual_geometry(image.width, image.height))
            has_image = image is not None
            pkey = (question, has_image)
            if pkey not in prompt_cache:
                prompt_cache[pkey] = prompt_hash(model, question, has_image)

            raw: str | None = None
            prediction = "[ERROR]"
            error_msg: str | None = None
            t_row = time.time()
            for attempt in range(args.max_retries + 1):
                try:
                    raw, prediction = run_model(model, question, image, task)
                    error_msg = None
                    break
                except Exception as exc:  # noqa: BLE001
                    error_msg = f"{type(exc).__name__}: {str(exc)[:500]}"
                    log(f"  row error ({t['question_id']}, {t['viz_type']}) "
                        f"attempt {attempt + 1}: {error_msg[:200]}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    time.sleep(2.0 * (attempt + 1))
            latency = time.time() - t_row

            bundle = compute_metrics(prediction=prediction, answer=str(item["answer"]))
            row: dict[str, Any] = {
                "question_id": item["question_id"],
                "question": item["question"],
                "answer": item["answer"],
                "prediction": prediction,
                "modality": item["modality"],
                "source": str(item.get("source", "synthetic")),
                "viz_type": t["viz_type"],
                "difficulty": item["difficulty"],
                "task": item["task"],
                "exact_match": bundle.exact,
                "f1": bundle.f1,
                "numeric_accuracy": bundle.numeric,
                "raw_response": raw,
                "model_id": model_id,
                "model_revision": revision,
                **img_meta,
                "prompt_sha256": prompt_cache[pkey],
                "suite": t["suite"],
                "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "latency_s": round(latency, 3),
            }
            if error_msg is not None:
                row["error"] = error_msg
                n_err += 1
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_new += 1

            if k % args.checkpoint_every == 0:
                flush()
                el = time.time() - run_start
                rate = n_new / el * 60 if el > 0 else 0.0
                eta_h = (len(pending) - k) / (n_new / el) / 3600 if n_new else 0.0
                mem = (torch.cuda.max_memory_allocated() / 2**30
                       if torch.cuda.is_available() else 0.0)
                log(f"{k}/{len(pending)} new={n_new} err={n_err} rate={rate:.1f} rows/min "
                    f"ETA={eta_h:.2f}h peak_alloc={mem:.1f}GiB")
    finally:
        flush()
        out.close()
        el = time.time() - run_start
        mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
        log(f"{'STOPPED' if shutdown else 'DONE'}: new={n_new} err={n_err} "
            f"elapsed={el:.1f}s rate={n_new / el * 60 if el else 0:.1f} rows/min "
            f"peak_alloc={mem:.1f}GiB peak_reserved="
            f"{(torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0):.1f}GiB")


if __name__ == "__main__":
    main()
