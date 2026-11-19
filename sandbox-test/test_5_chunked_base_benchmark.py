"""Test 5: Chunked 2-5s Speech Evaluation for faster-whisper/base.en + hotwords (CPU INT8).

Compares full-file transcription against 2-5s chunking strategies on CPU INT8 (no CUDA):
  1. Full-file baseline (reference WER, Term Hit, total latency)
  2. Fixed 2.5s slicing (fast UI trigger cadence, tests boundary cuts)
  3. Fixed 4.5s slicing (longer context window, still within 2-5s)
  4. Pause-aware 2.0-5.0s chunking + context tail carryover (splits on lowest-energy
     micro-pause inside [2.0s, 5.0s] and carries previous tail words into initial_prompt)

Measures per-chunk P50/P95/Max latency, RTF, RSS memory, WER, and meeting term hit rate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from test_1_stt_benchmark import (
    WhisperRunner,
    compute_wer,
    get_process_rss_mb,
    load_audio_mono_16k,
)
from test_3_long_audio_benchmark import LongSample, build_long_samples
from test_4_hotword_benchmark import (
    SAMPLE_HOTWORDS,
    build_prompt_string,
    compute_term_hit_rate,
)

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000
VN_SAMPLE_IDS = {"L05_20s", "L06_76s", "L07_33s"}


def slice_fixed_chunks(audio: np.ndarray, chunk_sec: float) -> list[np.ndarray]:
    """Slice 16kHz mono audio into contiguous fixed-duration chunks."""
    step = int(round(chunk_sec * SAMPLE_RATE))
    min_tail = int(0.25 * SAMPLE_RATE)
    chunks: list[np.ndarray] = []
    for start in range(0, len(audio), step):
        seg = audio[start : start + step]
        if len(seg) < min_tail and chunks:
            chunks[-1] = np.concatenate([chunks[-1], seg])
        elif len(seg) >= int(0.1 * SAMPLE_RATE):
            chunks.append(seg)
    return chunks


def slice_pause_aware_chunks(
    audio: np.ndarray,
    min_sec: float = 2.0,
    max_sec: float = 5.0,
    frame_ms: int = 80,
) -> list[np.ndarray]:
    """Split audio into [min_sec, max_sec] chunks at the quietest micro-pause frame."""
    total_len = len(audio)
    min_samples = int(round(min_sec * SAMPLE_RATE))
    max_samples = int(round(max_sec * SAMPLE_RATE))
    frame_len = max(1, int(round(frame_ms * SAMPLE_RATE / 1000.0)))

    chunks: list[np.ndarray] = []
    cursor = 0
    while cursor < total_len:
        remaining = total_len - cursor
        if remaining <= max_samples:
            seg = audio[cursor:]
            if len(seg) < int(0.4 * SAMPLE_RATE) and chunks:
                chunks[-1] = np.concatenate([chunks[-1], seg])
            else:
                chunks.append(seg)
            break

        search_start = cursor + min_samples
        search_end = min(total_len, cursor + max_samples)
        best_split = search_end
        best_rms = float("inf")

        for pos in range(search_start, search_end - frame_len + 1, frame_len):
            win = audio[pos : pos + frame_len]
            rms = float(np.sqrt(np.mean(win * win)))
            if rms < best_rms:
                best_rms = rms
                best_split = pos + frame_len // 2

        chunks.append(audio[cursor:best_split])
        cursor = best_split
    return chunks


def percentile_ms(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def evaluate_strategy(
    runner: WhisperRunner,
    samples: list[LongSample],
    strategy_name: str,
    mode: str,
    chunk_sec: float = 0.0,
    carry_context: bool = False,
) -> dict[str, Any]:
    """Run one chunking strategy across selected samples on an already-loaded runner."""
    rss_before = get_process_rss_mb()
    peak_rss = rss_before
    sample_rows: list[dict[str, Any]] = []

    for sample in samples:
        if not sample.wav_path.exists():
            continue

        terms = SAMPLE_HOTWORDS.get(sample.sample_id, [])
        base_prompt = build_prompt_string(terms) if terms else ""
        runner.hotwords = ", ".join(terms) if terms else None
        runner.initial_prompt = base_prompt or None

        audio = load_audio_mono_16k(sample.wav_path)
        duration_sec = len(audio) / SAMPLE_RATE

        if mode == "full":
            chunks = [audio]
        elif mode == "fixed":
            chunks = slice_fixed_chunks(audio, chunk_sec=chunk_sec)
        elif mode == "pause_aware":
            chunks = slice_pause_aware_chunks(audio, min_sec=2.0, max_sec=5.0)
        else:
            raise ValueError(f"Unsupported chunking mode: {mode}")

        chunk_details: list[dict[str, Any]] = []
        chunk_latencies: list[float] = []
        chunk_texts: list[str] = []
        prev_tail_words: list[str] = []

        t_total_start = time.perf_counter()
        for idx, chunk in enumerate(chunks, start=1):
            if carry_context and prev_tail_words:
                tail = " ".join(prev_tail_words[-10:])
                runner.initial_prompt = f"{base_prompt} {tail}".strip()
            else:
                runner.initial_prompt = base_prompt or None

            t0 = time.perf_counter()
            text = runner.transcribe(chunk, sample.wav_path)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            peak_rss = max(peak_rss, get_process_rss_mb())

            chunk_latencies.append(lat_ms)
            if text:
                chunk_texts.append(text)
                prev_tail_words.extend(text.split())
                prev_tail_words = prev_tail_words[-12:]

            chunk_details.append({
                "chunk_index": idx,
                "chunk_sec": round(len(chunk) / SAMPLE_RATE, 2),
                "latency_ms": round(lat_ms, 1),
                "text": text,
            })

        total_ms = (time.perf_counter() - t_total_start) * 1000.0
        merged_text = " ".join(chunk_texts).strip()
        hit_rate, matched_terms = compute_term_hit_rate(terms, merged_text)
        wer = compute_wer(sample.expected_text, merged_text)

        sample_rows.append({
            "sample_id": sample.sample_id,
            "duration_sec": round(duration_sec, 2),
            "chunk_count": len(chunks),
            "avg_chunk_sec": round(float(np.mean([len(c) / SAMPLE_RATE for c in chunks])), 2),
            "chunk_p50_ms": round(percentile_ms(chunk_latencies, 50), 1),
            "chunk_p95_ms": round(percentile_ms(chunk_latencies, 95), 1),
            "chunk_max_ms": round(max(chunk_latencies) if chunk_latencies else 0.0, 1),
            "total_latency_ms": round(total_ms, 1),
            "rtf": round((total_ms / 1000.0) / max(0.01, duration_sec), 3),
            "wer": round(wer, 4),
            "term_hit_rate": hit_rate,
            "matched_terms": matched_terms,
            "predicted_text": merged_text,
            "chunks": chunk_details,
        })

    return {
        "strategy": strategy_name,
        "engine": runner.name,
        "rss_delta_mb": round(max(0.0, peak_rss - rss_before), 1),
        "peak_rss_mb": round(peak_rss, 1),
        "samples": sample_rows,
    }


def write_chunked_report(summaries: list[dict[str, Any]], md_path: Path, json_path: Path) -> None:
    lines = [
        "# Báo cáo Kiểm thử Tách Chunk 2-5s — `faster-whisper/base.en + hotwords` (CPU INT8)",
        "",
        "- **Mục tiêu:** So sánh chạy full-file vs tách chunk 2-5 giây (`fixed` và `pause-aware 2-5s + context carryover`) trên CPU INT8 (bỏ qua CUDA).",
        "- **Ngưỡng đối chiếu:** Per-chunk P95 <= 500 ms (trần toàn pipeline), giữ WER & Term Hit không tụt so với full-file.",
        "",
        "## 1. Bảng tổng hợp hiệu năng & độ chính xác",
        "",
        "| Strategy | Sample | Chunks | Avg Chunk (s) | Chunk P50 (ms) | Chunk P95 (ms) | Chunk Max (ms) | Total (ms) | WER | Term Hit |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for strat in summaries:
        for r in strat["samples"]:
            lines.append(
                f"| {strat['strategy']} | {r['sample_id']} | {r['chunk_count']} | "
                f"{r['avg_chunk_sec']:.2f} | {r['chunk_p50_ms']} | {r['chunk_p95_ms']} | "
                f"{r['chunk_max_ms']} | {r['total_latency_ms']} | "
                f"{r['wer']*100:.1f}% | {r['term_hit_rate']*100:.1f}% |"
            )

    lines.extend(["", "## 2. Đối chiếu Transcript ghép từ các Chunk (Mẫu tiếng Việt L05-L07)", ""])
    for strat in summaries:
        vn_rows = [r for r in strat["samples"] if r["sample_id"] in VN_SAMPLE_IDS]
        if not vn_rows:
            continue
        lines.append(f"### {strat['strategy']}")
        for r in vn_rows:
            lines.append(f"- **{r['sample_id']}** (WER `{r['wer']*100:.1f}%`, Term Hit `{r['term_hit_rate']*100:.1f}%`, Chunk P95 `{r['chunk_p95_ms']} ms`): {r['predicted_text']}")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fluencieer Sandbox Test 5: 2-5s Chunking Benchmark for faster-whisper/base.en + hotwords (CPU INT8)"
    )
    parser.add_argument("--vn-only", action="store_true", help="Benchmark only Vietnamese-accented L05-L07 samples")
    parser.add_argument("--sample", default=None, help="Filter to a single sample_id (e.g. L06_76s)")
    parser.add_argument("--beam-size", type=int, default=2, help="Beam size for Whisper decoding (default: 2)")
    args = parser.parse_args()

    samples = build_long_samples()
    if args.vn_only:
        samples = [s for s in samples if s.sample_id in VN_SAMPLE_IDS]
    elif args.sample:
        samples = [s for s in samples if s.sample_id == args.sample]

    if not samples:
        raise SystemExit("No matching audio samples found.")

    runner = WhisperRunner(
        model_size="base.en",
        device="cpu",
        compute_type="int8",
        hotwords="active",
        beam_size=args.beam_size,
        vad_filter=True,
    )
    print(f"[Test 5] Loading {runner.name} (beam_size={args.beam_size}) ...")
    t_load = time.perf_counter()
    runner.load()
    print(f"[Test 5] Loaded in {(time.perf_counter() - t_load) * 1000.0:.1f} ms")

    strategies = [
        ("full-file (baseline)", "full", 0.0, False),
        ("fixed-chunk 2.5s", "fixed", 2.5, False),
        ("fixed-chunk 4.5s", "fixed", 4.5, False),
        ("pause-aware 2-5s + context", "pause_aware", 0.0, True),
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    try:
        for name, mode, chunk_sec, carry_context in strategies:
            print(f"[Test 5] Evaluating {name} ...")
            res = evaluate_strategy(
                runner=runner,
                samples=samples,
                strategy_name=name,
                mode=mode,
                chunk_sec=chunk_sec,
                carry_context=carry_context,
            )
            summaries.append(res)
            for row in res["samples"]:
                print(
                    f"  {row['sample_id']}: chunks={row['chunk_count']} "
                    f"p50={row['chunk_p50_ms']}ms p95={row['chunk_p95_ms']}ms "
                    f"wer={row['wer']*100:.1f}% term_hit={row['term_hit_rate']*100:.1f}%"
                )
    finally:
        runner.unload()

    md_out = RESULTS_DIR / "chunked_base_report.md"
    json_out = RESULTS_DIR / "chunked_base_results.json"
    write_chunked_report(summaries, md_out, json_out)
    print(f"[Done] Saved chunked base.en reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
