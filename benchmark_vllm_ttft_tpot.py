#!/usr/bin/env python3
"""Benchmark vLLM TTFT and TPOT across long context lengths.

Metrics:
- TTFT: time from request start to first non-empty streamed text chunk.
- TPOT excl first token: (total_latency - ttft) / (generated_tokens - 1).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import requests
from transformers import AutoTokenizer


DEFAULT_LENGTHS = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 98304]
max_model_len = 524288  # to support up to 512K prompt + 100 gen tokens with some margin


@dataclass
class RunResult:
    prompt_tokens: int
    run_index: int
    ok: bool
    error: str
    ttft_s: Optional[float]
    total_latency_s: Optional[float]
    generated_tokens: int
    tpot_excl_first_s: Optional[float]
    attempt: int


def parse_length_token(text: str) -> int:
    token = text.strip().lower()
    if not token:
        raise ValueError("empty length token")
    if token.endswith("k"):
        prompt_len = int(float(token[:-1]) * 1024)
        if prompt_len >= max_model_len:
            prompt_len -= 128  # reserve some tokens for generation
        if prompt_len > max_model_len:
            raise ValueError("length exceeds max_model_len")
        return prompt_len
    if token.endswith("m"):
        # raise ValueError("length in millions is not supported due to vLLM max_model_len limit")
        return int(float(token[:-1]) * 1024 * 1024 - 128)
    return int(token)


def parse_lengths(lengths_arg: str) -> List[int]:
    parts = [p.strip() for p in lengths_arg.split(",") if p.strip()]
    if not parts:
        raise ValueError("length list is empty")
    vals = sorted({parse_length_token(p) for p in parts})
    if vals[0] <= 0:
        raise ValueError("all lengths must be > 0")
    return vals


def percentile(values: List[float], q: float) -> Optional[float]:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    rank = (len(values) - 1) * q
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return values[low]
    frac = rank - low
    return values[low] * (1.0 - frac) + values[high] * frac


def _expand_token_ids(seed_ids: List[int], target_len: int) -> List[int]:
    times = (target_len // len(seed_ids)) + 2
    return (seed_ids * times)[:target_len]


def build_prompt_with_exact_tokens(tokenizer: AutoTokenizer, target_tokens: int) -> Tuple[str, int]:
    """Build a prompt whose tokenized length is as close as possible to target.

    We prefer exact match and run a small correction loop if decode/encode changes length.
    """
    seed_ids = tokenizer.encode(" hello", add_special_tokens=False)
    if not seed_ids:
        seed_ids = tokenizer.encode("a", add_special_tokens=False)
    if not seed_ids:
        raise RuntimeError("failed to obtain seed token ids from tokenizer")

    token_ids = _expand_token_ids(seed_ids, target_tokens)
    prompt = tokenizer.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    for _ in range(8):
        cur_len = len(tokenizer.encode(prompt, add_special_tokens=False))
        if cur_len == target_tokens:
            return prompt, cur_len
        if cur_len < target_tokens:
            prompt += " a" * (target_tokens - cur_len)
        else:
            ids = tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens]
            prompt = tokenizer.decode(
                ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

    final_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if final_len < target_tokens:
        prompt += " a" * (target_tokens - final_len)
        final_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    if final_len > target_tokens:
        ids = tokenizer.encode(prompt, add_special_tokens=False)[:target_tokens]
        prompt = tokenizer.decode(
            ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        final_len = len(tokenizer.encode(prompt, add_special_tokens=False))
    return prompt, final_len


def stream_completion(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    request_timeout_s: int,
    min_tokens: Optional[int] = None,
    ignore_eos: bool = False,
) -> Tuple[float, float, str, Optional[int]]:
    """Send one streaming completion request and return timing + generated text."""
    url = base_url.rstrip("/") + "/v1/completions"
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if min_tokens is not None:
        payload["min_tokens"] = min_tokens
    if ignore_eos:
        payload["ignore_eos"] = True

    first_token_at: Optional[float] = None
    chunks: List[str] = []
    usage_completion_tokens: Optional[int] = None
    start = time.perf_counter()

    with requests.post(
        url,
        json=payload,
        stream=True,
        timeout=(10, request_timeout_s),
    ) as resp:
        resp.raise_for_status()
        for raw in resp.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            obj = json.loads(data)
            usage = obj.get("usage")
            if isinstance(usage, dict) and usage.get("completion_tokens") is not None:
                usage_completion_tokens = int(usage["completion_tokens"])
            text_piece = ""
            choices = obj.get("choices")
            if choices and isinstance(choices, list):
                text_piece = choices[0].get("text") or ""
            if text_piece:
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                chunks.append(text_piece)

    end = time.perf_counter()
    if first_token_at is None:
        raise RuntimeError("no streamed text token received")
    ttft_s = first_token_at - start
    total_latency_s = end - start
    return ttft_s, total_latency_s, "".join(chunks), usage_completion_tokens


def run_one(
    base_url: str,
    model: str,
    prompt: str,
    prompt_tokens: int,
    run_index: int,
    gen_tokens: int,
    request_timeout_s: int,
    retry: int,
    tokenizer: AutoTokenizer,
    force_min_tokens: bool,
    ignore_eos: bool,
) -> RunResult:
    last_err = ""
    for attempt in range(1, retry + 2):
        try:
            ttft_s, total_latency_s, generated_text, usage_completion_tokens = stream_completion(
                base_url=base_url,
                model=model,
                prompt=prompt,
                max_tokens=gen_tokens,
                request_timeout_s=request_timeout_s,
                min_tokens=gen_tokens if force_min_tokens else None,
                ignore_eos=ignore_eos,
            )
            generated_tokens = (
                usage_completion_tokens
                if usage_completion_tokens is not None
                else len(tokenizer.encode(generated_text, add_special_tokens=False))
            )
            if generated_tokens < 2:
                return RunResult(
                    prompt_tokens=prompt_tokens,
                    run_index=run_index,
                    ok=False,
                    error=f"generated_tokens={generated_tokens}, need >=2",
                    ttft_s=ttft_s,
                    total_latency_s=total_latency_s,
                    generated_tokens=generated_tokens,
                    tpot_excl_first_s=None,
                    attempt=attempt,
                )
            tpot = (total_latency_s - ttft_s) / float(generated_tokens - 1)
            return RunResult(
                prompt_tokens=prompt_tokens,
                run_index=run_index,
                ok=True,
                error="",
                ttft_s=ttft_s,
                total_latency_s=total_latency_s,
                generated_tokens=generated_tokens,
                tpot_excl_first_s=tpot,
                attempt=attempt,
            )
        except Exception as exc:  # noqa: BLE001
            last_err = str(exc)
            if attempt <= retry:
                time.sleep(1.0)
            continue

    return RunResult(
        prompt_tokens=prompt_tokens,
        run_index=run_index,
        ok=False,
        error=last_err or "unknown error",
        ttft_s=None,
        total_latency_s=None,
        generated_tokens=0,
        tpot_excl_first_s=None,
        attempt=retry + 1,
    )


def summarize_length(results: Iterable[RunResult]) -> Dict[str, Optional[float]]:
    rows = list(results)
    valid = [r for r in rows if r.ok]
    ttft_vals = sorted([r.ttft_s for r in valid if r.ttft_s is not None])
    tpot_vals = sorted([r.tpot_excl_first_s for r in valid if r.tpot_excl_first_s is not None])
    total_vals = sorted([r.total_latency_s for r in valid if r.total_latency_s is not None])
    return {
        "runs_total": len(rows),
        "runs_valid": len(valid),
        "runs_failed": len(rows) - len(valid),
        "ttft_mean_s": statistics.mean(ttft_vals) if ttft_vals else None,
        "ttft_p50_s": percentile(ttft_vals, 0.5) if ttft_vals else None,
        "ttft_p90_s": percentile(ttft_vals, 0.9) if ttft_vals else None,
        "tpot_excl_first_mean_s": statistics.mean(tpot_vals) if tpot_vals else None,
        "tpot_excl_first_p50_s": percentile(tpot_vals, 0.5) if tpot_vals else None,
        "tpot_excl_first_p90_s": percentile(tpot_vals, 0.9) if tpot_vals else None,
        "total_latency_mean_s": statistics.mean(total_vals) if total_vals else None,
    }


def format_s(val: Optional[float]) -> str:
    if val is None:
        return "NA"
    return f"{val:.4f}"


def print_summary_table(summary_rows: List[Dict[str, object]]) -> None:
    header = (
        "len_tok | valid/total | ttft_mean(s) | ttft_p50(s) | ttft_p90(s) | "
        "tpot_mean_excl_first(s) | tpot_p50(s) | tpot_p90(s)"
    )
    print("\n" + header)
    print("-" * len(header))
    for row in summary_rows:
        print(
            f"{row['prompt_tokens']:>7} | "
            f"{row['runs_valid']:>2}/{row['runs_total']:<2} | "
            f"{format_s(row['ttft_mean_s']):>11} | "
            f"{format_s(row['ttft_p50_s']):>11} | "
            f"{format_s(row['ttft_p90_s']):>11} | "
            f"{format_s(row['tpot_excl_first_mean_s']):>22} | "
            f"{format_s(row['tpot_excl_first_p50_s']):>11} | "
            f"{format_s(row['tpot_excl_first_p90_s']):>11}"
        )


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "prompt_tokens",
        "actual_prompt_tokens",
        "runs_total",
        "runs_valid",
        "runs_failed",
        "ttft_mean_s",
        "ttft_p50_s",
        "ttft_p90_s",
        "tpot_excl_first_mean_s",
        "tpot_excl_first_p50_s",
        "tpot_excl_first_p90_s",
        "total_latency_mean_s",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM TTFT and TPOT (excluding first token)."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8711", help="vLLM base URL")
    parser.add_argument("--model", required=True, help="served model name")
    parser.add_argument(
        "--model-path",
        required=True,
        help="local model path for tokenizer loading",
    )
    parser.add_argument(
        "--lengths",
        default="1k,2k,4k,8k,16k,32k,64k,128k,256k,512k",
        help="comma separated prompt lengths in tokens",
    )
    parser.add_argument("--gen-tokens", type=int, default=100, help="completion token target")
    parser.add_argument("--warmup-runs", type=int, default=2, help="warmup request count")
    parser.add_argument("--runs-per-len", type=int, default=3, help="runs per prompt length")
    parser.add_argument("--retry", type=int, default=1, help="retry count per run on failure")
    parser.add_argument("--request-timeout-s", type=int, default=5000, help="request timeout")
    parser.add_argument(
        "--force-min-tokens",
        action="store_true",
        help="set min_tokens=max_tokens to avoid early stop",
    )
    parser.add_argument(
        "--ignore-eos",
        action="store_true",
        help="set ignore_eos=true to avoid EOS-based early stop",
    )
    parser.add_argument(
        "--output",
        default="",
        help="optional output json path; csv uses same basename",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lengths = parse_lengths(args.lengths) if args.lengths else DEFAULT_LENGTHS

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output:
        json_path = Path(args.output).expanduser().resolve()
    else:
        out_dir = Path("benchmark_results").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / f"ttft_tpot_{timestamp}.json"
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = json_path.with_suffix(".csv")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    prompts: Dict[int, Tuple[str, int]] = {}

    print("Building prompts by token length ...")
    for length in lengths:
        prompt, actual_len = build_prompt_with_exact_tokens(tokenizer, length)
        prompts[length] = (prompt, actual_len)
        print(f"  requested={length}, actual={actual_len}")

    warmup_len = lengths[0]
    warmup_prompt, warmup_actual = prompts[warmup_len]
    print(f"\nWarmup: {args.warmup_runs} runs at {warmup_len} tokens (actual={warmup_actual})")
    for idx in range(args.warmup_runs):
        try:
            _ = stream_completion(
                base_url=args.base_url,
                model=args.model,
                prompt=warmup_prompt,
                max_tokens=min(32, args.gen_tokens),
                request_timeout_s=args.request_timeout_s,
                min_tokens=None,
                ignore_eos=False,
            )
            print(f"  warmup {idx + 1}/{args.warmup_runs}: ok")
        except Exception as exc:  # noqa: BLE001
            print(f"  warmup {idx + 1}/{args.warmup_runs}: failed ({exc})")

    print("\nBenchmarking ...")
    all_results: List[RunResult] = []
    summary_rows: List[Dict[str, object]] = []
    for req_len in lengths:
        prompt, actual_len = prompts[req_len]
        length_results: List[RunResult] = []
        print(f"\nLength {req_len} (actual={actual_len}), runs={args.runs_per_len}")
        for run_idx in range(1, args.runs_per_len + 1):
            result = run_one(
                base_url=args.base_url,
                model=args.model,
                prompt=prompt,
                prompt_tokens=req_len,
                run_index=run_idx,
                gen_tokens=args.gen_tokens,
                request_timeout_s=args.request_timeout_s,
                retry=args.retry,
                tokenizer=tokenizer,
                force_min_tokens=args.force_min_tokens,
                ignore_eos=args.ignore_eos,
            )
            length_results.append(result)
            all_results.append(result)
            if result.ok:
                print(
                    "  run "
                    f"{run_idx}/{args.runs_per_len}: "
                    f"ttft={format_s(result.ttft_s)}s, "
                    f"total={format_s(result.total_latency_s)}s, "
                    f"gen={result.generated_tokens}, "
                    f"tpot_excl_first={format_s(result.tpot_excl_first_s)}s"
                )
            else:
                print(
                    f"  run {run_idx}/{args.runs_per_len}: failed "
                    f"(attempt={result.attempt}, err={result.error})"
                )
        agg = summarize_length(length_results)
        agg_row: Dict[str, object] = {
            "prompt_tokens": req_len,
            "actual_prompt_tokens": actual_len,
            **agg,
        }
        summary_rows.append(agg_row)

    print_summary_table(summary_rows)

    payload = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "model_path": args.model_path,
            "gen_tokens": args.gen_tokens,
            "warmup_runs": args.warmup_runs,
            "runs_per_len": args.runs_per_len,
            "retry": args.retry,
            "request_timeout_s": args.request_timeout_s,
            "force_min_tokens": args.force_min_tokens,
            "ignore_eos": args.ignore_eos,
            "lengths_requested": lengths,
            "timestamp": timestamp,
            "note": (
                "To benchmark up to 128K, vLLM server max_model_len should be >= 131072."
            ),
        },
        "summary": summary_rows,
        "runs": [asdict(r) for r in all_results],
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    write_csv(csv_path, summary_rows)

    print(f"\nSaved summary JSON: {json_path}")
    print(f"Saved summary CSV : {csv_path}")


if __name__ == "__main__":
    main()

"""
python benchmark_vllm_ttft_tpot.py \
  --base-url http://127.0.0.1:8711 \
  --model SSE_SWA_MOBA \
  --model-path /mnt/jfzn/pyq/ColossalAI-dev/checkpoints/sse_swa128_drop0p5_moba4k_top12_4b_lr5en6_bsz32_pt69p86_ct512k5btk_sft500k_rsft500k_24k/modeling \
  --runs-per-len 20 \
  --warmup-runs 3 \
  --gen-tokens 100 \
  --force-min-tokens \
  --lengths 4m \
  --ignore-eos

python benchmark_vllm_ttft_tpot.py \
  --base-url http://127.0.0.1:8711 \
  --model Qwen \
  --model-path /mnt/jfzn/models/Qwen3-4B-Thinking-2507 \
  --runs-per-len 1 \
  --warmup-runs 1 \
  --gen-tokens 100 \
  --force-min-tokens \
  --lengths 1m,2m,4m \
  --ignore-eos

"""