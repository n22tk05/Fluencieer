"""Test 6: Sherpa-ONNX Streaming Zipformer Comprehensive Benchmark (CPU).

Evaluates streaming Zipformer transducer models across ALL Fluencieer test dimensions:
  - Test 1/2 Streaming Cadence: 200ms Web-Audio-style chunk streaming decode P50/P95/Max (ms) & flush time
  - Test 2 Memory Cap: Process RSS Delta & Peak RSS vs 1.6 GB PRD ceiling
  - Test 3 Long-Audio Scaling: 6s -> 75s clips (L01-L04, H01, L05-L07) WER, CER, RTF, zero truncation
  - Test 4 Hotword Biasing: greedy_search (baseline) vs modified_beam_search + hotwords_file on L05-L07

Automatically downloads the HuggingFace ONNX model repo once and exports bpe.vocab from bpe.model.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from huggingface_hub import snapshot_download
import numpy as np
import sentencepiece as spm
import sherpa_onnx

from test_1_stt_benchmark import (
    compute_cer,
    compute_wer,
    get_process_rss_mb,
    load_audio_mono_16k,
)
from test_3_long_audio_benchmark import LongSample, build_long_samples
from test_4_hotword_benchmark import SAMPLE_HOTWORDS, compute_term_hit_rate

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000
DEFAULT_MODEL_REPO = "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-06-21"


def find_onnx_triplet(model_dir: Path, precision: str) -> tuple[Path, Path, Path, Path]:
    """Locate tokens.txt and encoder/decoder/joiner ONNX files for int8 or fp32."""
    tokens = model_dir / "tokens.txt"
    if not tokens.exists():
        raise FileNotFoundError(f"Missing tokens.txt in {model_dir}")

    def pick(prefix: str) -> Path:
        int8_matches = sorted(model_dir.glob(f"{prefix}-*.int8.onnx"))
        fp32_matches = [
            p for p in sorted(model_dir.glob(f"{prefix}-*.onnx")) if not p.name.endswith(".int8.onnx")
        ]
        if precision == "int8":
            if int8_matches:
                return int8_matches[0]
            if fp32_matches:
                return fp32_matches[0]
        else:
            if fp32_matches:
                return fp32_matches[0]
            if int8_matches:
                return int8_matches[0]
        raise FileNotFoundError(f"Missing {prefix}-*.onnx ({precision}) in {model_dir}")

    return tokens, pick("encoder"), pick("decoder"), pick("joiner")


def ensure_bpe_vocab(model_dir: Path) -> Path:
    """Ensure bpe.vocab exists (exporting from bpe.model via sentencepiece if absent)."""
    vocab_path = model_dir / "bpe.vocab"
    if vocab_path.exists():
        return vocab_path

    bpe_model = model_dir / "bpe.model"
    if not bpe_model.exists():
        candidates = sorted(model_dir.glob("*.model"))
        if candidates:
            bpe_model = candidates[0]

    lines: list[str] = []
    if bpe_model.exists():
        sp = spm.SentencePieceProcessor()
        sp.load(str(bpe_model))
        for idx in range(sp.get_piece_size()):
            lines.append(f"{sp.id_to_piece(idx)} {sp.get_score(idx)}")
    else:
        tokens_file = model_dir / "tokens.txt"
        for raw_line in tokens_file.read_text(encoding="utf-8").splitlines():
            parts = raw_line.strip().split()
            if parts:
                lines.append(f"{parts[0]} -1.0")

    vocab_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return vocab_path


def build_hotwords_text(samples: list[LongSample]) -> str:
    """Create uppercase BPE-compatible meeting glossary lines for sherpa-onnx hotwords_file."""
    seen: set[str] = set()
    lines: list[str] = []
    for sample in samples:
        for term in SAMPLE_HOTWORDS.get(sample.sample_id, []):
            cleaned = "".join(ch if (ch.isalpha() or ch in "' ") else " " for ch in term.upper())
            words = [w for w in cleaned.split() if len(w) >= 2 or w in {"A", "I"}]
            phrase = " ".join(words).strip()
            if phrase and phrase not in seen:
                seen.add(phrase)
                lines.append(phrase)
    return "\n".join(lines) + "\n"


def benchmark_sherpa_config(
    model_dir: Path,
    bpe_vocab: Path,
    samples: list[LongSample],
    precision: str,
    use_hotwords: bool,
    hotwords_file: Path | None,
    hotwords_score: float,
    num_threads: int,
    chunk_ms: int,
) -> dict[str, Any]:
    """Load one Sherpa-ONNX config and stream 200ms chunks across all samples."""
    tokens, encoder, decoder, joiner = find_onnx_triplet(model_dir, precision=precision)
    decoding_method = "modified_beam_search" if use_hotwords else "greedy_search"
    label = f"sherpa-onnx/zipformer-en ({precision}, {'beam+hotwords' if use_hotwords else 'greedy'}, {chunk_ms}ms)"

    rss_before = get_process_rss_mb()
    t_load = time.perf_counter()
    kwargs: dict[str, Any] = {
        "tokens": str(tokens),
        "encoder": str(encoder),
        "decoder": str(decoder),
        "joiner": str(joiner),
        "num_threads": num_threads,
        "sample_rate": SAMPLE_RATE,
        "feature_dim": 80,
        "decoding_method": decoding_method,
        "max_active_paths": 4,
        "provider": "cpu",
    }
    if use_hotwords and hotwords_file is not None:
        kwargs.update({
            "hotwords_file": str(hotwords_file),
            "hotwords_score": hotwords_score,
            "modeling_unit": "bpe",
            "bpe_vocab": str(bpe_vocab),
        })

    recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(**kwargs)

    # Warmup stream with 1s silence to stabilize ONNX Runtime session arenas
    warmup_stream = recognizer.create_stream()
    warmup_stream.accept_waveform(SAMPLE_RATE, np.zeros(SAMPLE_RATE, dtype=np.float32))
    warmup_stream.input_finished()
    while recognizer.is_ready(warmup_stream):
        recognizer.decode_stream(warmup_stream)
    load_ms = (time.perf_counter() - t_load) * 1000.0
    peak_rss = get_process_rss_mb()

    chunk_samples = max(160, int(SAMPLE_RATE * chunk_ms / 1000))
    tail_pad = np.zeros(int(0.4 * SAMPLE_RATE), dtype=np.float32)

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.wav_path.exists():
            continue

        audio = load_audio_mono_16k(sample.wav_path)
        duration_sec = len(audio) / SAMPLE_RATE
        stream = recognizer.create_stream()

        chunk_latencies: list[float] = []
        t_total_start = time.perf_counter()

        for start in range(0, len(audio), chunk_samples):
            frame = audio[start : start + chunk_samples]
            t0 = time.perf_counter()
            stream.accept_waveform(SAMPLE_RATE, frame)
            while recognizer.is_ready(stream):
                recognizer.decode_stream(stream)
            _ = recognizer.get_result(stream)
            chunk_latencies.append((time.perf_counter() - t0) * 1000.0)

        t_flush = time.perf_counter()
        stream.accept_waveform(SAMPLE_RATE, tail_pad)
        stream.input_finished()
        while recognizer.is_ready(stream):
            recognizer.decode_stream(stream)
        pred = recognizer.get_result(stream).strip()
        flush_ms = (time.perf_counter() - t_flush) * 1000.0
        total_ms = (time.perf_counter() - t_total_start) * 1000.0

        peak_rss = max(peak_rss, get_process_rss_mb())
        terms = SAMPLE_HOTWORDS.get(sample.sample_id, [])
        hit_rate, matched_terms = compute_term_hit_rate(terms, pred)

        lat_arr = np.asarray(chunk_latencies, dtype=np.float64)
        rows.append({
            "sample_id": sample.sample_id,
            "duration_sec": round(duration_sec, 2),
            "chunk_count": len(chunk_latencies),
            "chunk_p50_ms": round(float(np.percentile(lat_arr, 50)), 2) if len(lat_arr) else 0.0,
            "chunk_p95_ms": round(float(np.percentile(lat_arr, 95)), 2) if len(lat_arr) else 0.0,
            "chunk_max_ms": round(float(np.max(lat_arr)), 2) if len(lat_arr) else 0.0,
            "flush_final_ms": round(flush_ms, 2),
            "total_decode_ms": round(total_ms, 1),
            "rtf": round((total_ms / 1000.0) / max(0.01, duration_sec), 4),
            "wer": round(compute_wer(sample.expected_text, pred), 4),
            "cer": round(compute_cer(sample.expected_text, pred), 4),
            "term_hit_rate": hit_rate,
            "matched_terms": matched_terms,
            "predicted_text": pred,
        })

    del recognizer
    gc.collect()

    return {
        "engine": label,
        "load_ms": round(load_ms, 1),
        "rss_delta_mb": round(max(0.0, peak_rss - rss_before), 1),
        "peak_rss_mb": round(peak_rss, 1),
        "samples": rows,
    }


def write_sherpa_report(
    summaries: list[dict[str, Any]],
    model_repo: str,
    md_path: Path,
    json_path: Path,
) -> None:
    lines = [
        "# Báo cáo Kiểm thử Toàn diện — Sherpa-ONNX Streaming Zipformer (CPU)",
        "",
        f"- **Model Repo:** `{model_repo}`",
        "- **Chế độ kiểm thử:** Streaming từng frame `200 ms`, đo P50/P95/Max per-frame, RTF, RAM RSS, WER/CER (L01-L07 + H01), và Hotwords (`modified_beam_search`).",
        "",
        "## 1. Tổng hợp Streaming Latency, RAM & Độ chính xác",
        "",
        "| Engine | RSS Delta (MB) | Peak RSS (MB) | Sample | Chunk P50 (ms) | Chunk P95 (ms) | Flush (ms) | Total (ms) | RTF | WER | Term Hit |",
        "| :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        for r in s["samples"]:
            lines.append(
                f"| {s['engine']} | {s['rss_delta_mb']} | {s['peak_rss_mb']} | {r['sample_id']} | "
                f"{r['chunk_p50_ms']} | {r['chunk_p95_ms']} | {r['flush_final_ms']} | "
                f"{r['total_decode_ms']} | {r['rtf']} | {r['wer']*100:.1f}% | {r['term_hit_rate']*100:.1f}% |"
            )

    lines.extend(["", "## 2. Transcript thực tế trên giọng Việt (L05-L07) & L06", ""])
    for s in summaries:
        lines.append(f"### {s['engine']}")
        for r in s["samples"]:
            if r["sample_id"] in {"L05_20s", "L06_76s", "L07_33s", "L01_6s", "H01_10s_human"}:
                matched = ", ".join(r["matched_terms"]) if r["matched_terms"] else "(none)"
                lines.append(
                    f"- **{r['sample_id']}** (WER `{r['wer']*100:.1f}%`, Term Hit `{r['term_hit_rate']*100:.1f}%`, "
                    f"Chunk P95 `{r['chunk_p95_ms']} ms`, Khớp: `{matched}`): {r['predicted_text']}"
                )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 6: Sherpa-ONNX Full Benchmark Suite")
    parser.add_argument("--model-repo", default=DEFAULT_MODEL_REPO, help="HuggingFace repo ID for streaming Zipformer")
    parser.add_argument("--threads", type=int, default=4, help="ONNX Runtime CPU threads (default: 4)")
    parser.add_argument("--chunk-ms", type=int, default=200, help="Streaming frame size in ms (default: 200)")
    parser.add_argument("--hotwords-score", type=float, default=2.0, help="Hotword biasing score (default: 2.0)")
    parser.add_argument("--sample", default=None, help="Optional single sample_id filter")
    args = parser.parse_args()

    samples = build_long_samples()
    if args.sample:
        samples = [s for s in samples if s.sample_id == args.sample]
    if not samples:
        raise SystemExit("No matching samples found.")

    print(f"[Test 6] Ensuring HuggingFace model {args.model_repo} is cached ...")
    model_dir = Path(snapshot_download(repo_id=args.model_repo))
    bpe_vocab = ensure_bpe_vocab(model_dir)

    hotwords_text = build_hotwords_text(samples)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    with tempfile.NamedTemporaryFile("w", suffix=".txt", encoding="utf-8", delete=False) as tf:
        tf.write(hotwords_text)
        hw_path = Path(tf.name)

    try:
        configs = [
            ("int8", False, None),
            ("int8", True, hw_path),
            ("fp32", True, hw_path),
        ]
        for precision, use_hw, hw_file in configs:
            print(f"[Test 6] Running precision={precision}, hotwords={use_hw}, chunk={args.chunk_ms}ms ...")
            try:
                res = benchmark_sherpa_config(
                    model_dir=model_dir,
                    bpe_vocab=bpe_vocab,
                    samples=samples,
                    precision=precision,
                    use_hotwords=use_hw,
                    hotwords_file=hw_file,
                    hotwords_score=args.hotwords_score,
                    num_threads=args.threads,
                    chunk_ms=args.chunk_ms,
                )
                summaries.append(res)
                for row in res["samples"]:
                    print(
                        f"  {row['sample_id']}: p50={row['chunk_p50_ms']}ms p95={row['chunk_p95_ms']}ms "
                        f"flush={row['flush_final_ms']}ms rtf={row['rtf']} "
                        f"wer={row['wer']*100:.1f}% term_hit={row['term_hit_rate']*100:.1f}%"
                    )
            except Exception as exc:
                print(f"  ! Skipped ({precision}, hotwords={use_hw}): {exc}")
    finally:
        hw_path.unlink(missing_ok=True)

    md_out = RESULTS_DIR / "sherpa_onnx_report.md"
    json_out = RESULTS_DIR / "sherpa_onnx_results.json"
    write_sherpa_report(summaries, args.model_repo, md_out, json_out)
    print(f"[Done] Saved Sherpa-ONNX reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
