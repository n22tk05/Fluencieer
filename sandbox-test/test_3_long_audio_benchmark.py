"""Test 3: Long-Audio Scaling Benchmark (Moonshine vs Faster-Whisper tiny.en).

Measures latency scaling, Real-Time Factor (RTF), RSS RAM, WER/CER, and output
completeness across 6s, 10s (human), 11s, 26s, and 41s+ English speech clips.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import moonshine_onnx
import psutil

from test_1_stt_benchmark import (
    MoonshineRunner,
    WhisperRunner,
    compute_cer,
    compute_wer,
    get_gpu_vram_mb,
    get_process_rss_mb,
    load_audio_mono_16k,
)

SANDBOX_DIR = Path(__file__).resolve().parent
LONG_DIR = SANDBOX_DIR / "audio_samples" / "long"
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000


@dataclass
class LongSample:
    sample_id: str
    wav_path: Path
    expected_text: str
    source_type: str


def build_long_samples() -> list[LongSample]:
    return [
        LongSample(
            sample_id="L01_6s",
            wav_path=LONG_DIR / "L01_5s.wav",
            expected_text=(
                "Regarding the backend timeline, we need to resolve the current "
                "database bottleneck before Friday."
            ),
            source_type="TTS en-US (David)",
        ),
        LongSample(
            sample_id="H01_10s_human",
            wav_path=moonshine_onnx.ASSETS_DIR / "beckett.wav",
            expected_text=(
                "Ever tried. Ever failed. No matter. Try again. Fail again. Fail better."
            ),
            source_type="Human speech (beckett.wav)",
        ),
        LongSample(
            sample_id="L02_11s",
            wav_path=LONG_DIR / "L02_10s.wav",
            expected_text=(
                "From a technical perspective, decoupling the payment queue reduces "
                "latency spikes during peak traffic. Next week we will review the "
                "deployment checklist with all stakeholders."
            ),
            source_type="TTS en-US (Zira)",
        ),
        LongSample(
            sample_id="L03_26s",
            wav_path=LONG_DIR / "L03_20s.wav",
            expected_text=(
                "Let me share a quick update on our sprint progress. We finished "
                "profiling the local inference pipeline and verified that our sliding "
                "window keeps memory usage stable below the target threshold. However, "
                "we noticed minor latency fluctuations when multiple browser tabs run "
                "WebRTC calls simultaneously, so we are preparing a rollback strategy "
                "and finalizing the acceptance criteria before our release demo."
            ),
            source_type="TTS en-US (David)",
        ),
        LongSample(
            sample_id="L04_42s",
            wav_path=LONG_DIR / "L04_35s.wav",
            expected_text=(
                "Before we lock the architecture for the prototype, I want to walk "
                "through three critical risks we discovered during testing. First, if "
                "the local speech recognition buffer grows unbounded over a sixty minute "
                "meeting, memory consumption can spike unexpectedly and degrade video "
                "call frame rates. Second, network dependent speech services introduce "
                "unpredictable delays above five hundred milliseconds, breaking the "
                "natural conversational rhythm. Third, speakers often pause briefly mid "
                "sentence while searching for precise domain terminology, so keeping "
                "prefix pinning active ensures the model prioritizes project vocabulary "
                "without duplicating context memory."
            ),
            source_type="TTS en-US (Zira)",
        ),
    ]


def benchmark_long_runner(runner: Any, samples: list[LongSample]) -> dict[str, Any]:
    """Load runner once and benchmark across all long audio files."""
    rss_before = get_process_rss_mb()
    vram_before = get_gpu_vram_mb()

    t_load = time.perf_counter()
    runner.load()
    load_ms = (time.perf_counter() - t_load) * 1000.0

    peak_rss = get_process_rss_mb()
    vram_after = get_gpu_vram_mb()
    rows: list[dict[str, Any]] = []

    for sample in samples:
        if not sample.wav_path.exists():
            continue
        audio = load_audio_mono_16k(sample.wav_path)
        duration_sec = len(audio) / SAMPLE_RATE

        # Run twice and take faster warm latency (second pass avoids cold page faults)
        latencies: list[float] = []
        pred = ""
        for _ in range(2):
            t0 = time.perf_counter()
            pred = runner.transcribe(audio, sample.wav_path)
            latencies.append((time.perf_counter() - t0) * 1000.0)
            peak_rss = max(peak_rss, get_process_rss_mb())

        best_ms = min(latencies)
        rtf = (best_ms / 1000.0) / duration_sec
        exp_words = len(sample.expected_text.split())
        pred_words = len(pred.split())
        rows.append({
            "sample_id": sample.sample_id,
            "source_type": sample.source_type,
            "duration_sec": round(duration_sec, 2),
            "latency_ms": round(best_ms, 1),
            "all_runs_ms": [round(x, 1) for x in latencies],
            "rtf": round(rtf, 4),
            "wer": round(compute_wer(sample.expected_text, pred), 4),
            "cer": round(compute_cer(sample.expected_text, pred), 4),
            "expected_words": exp_words,
            "predicted_words": pred_words,
            "predicted_text": pred,
        })

    runner.unload()
    return {
        "engine": runner.name,
        "load_ms": round(load_ms, 1),
        "rss_delta_mb": round(max(0.0, peak_rss - rss_before), 1),
        "peak_rss_mb": round(peak_rss, 1),
        "vram_delta_mb": (
            round(max(0.0, vram_after - vram_before), 1)
            if vram_before is not None and vram_after is not None
            else None
        ),
        "samples": rows,
    }


def write_long_report(summaries: list[dict[str, Any]], md_path: Path, json_path: Path) -> None:
    lines = [
        "# Báo cáo Test Đoạn Dài (6s – 42s) — Moonshine vs Faster-Whisper tiny.en",
        "",
        "| Engine | RSS Delta (MB) | Sample | Thời lượng (s) | Latency (ms) | RTF | WER | Từ (Pred/Exp) |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | ---: | :---: |",
    ]
    for summary in summaries:
        for row in summary["samples"]:
            lines.append(
                f"| {summary['engine']} | {summary['rss_delta_mb']} | {row['sample_id']} | "
                f"{row['duration_sec']} | {row['latency_ms']} | {row['rtf']} | "
                f"{row['wer']*100:.1f}% | {row['predicted_words']}/{row['expected_words']} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 3: Long-Audio Scaling Benchmark")
    parser.add_argument(
        "--include-cuda",
        action="store_true",
        help="Also attempt faster-whisper tiny.en on CUDA FP16 if CUDA libs are available",
    )
    args = parser.parse_args()

    samples = build_long_samples()
    runners: list[Any] = [
        MoonshineRunner("moonshine/tiny", precision="quantized"),
        MoonshineRunner("moonshine/tiny", precision="float"),
        MoonshineRunner("moonshine/base", precision="quantized"),
        WhisperRunner("tiny.en", device="cpu", compute_type="int8"),
    ]
    if args.include_cuda:
        runners.append(WhisperRunner("tiny.en", device="cuda", compute_type="float16"))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for runner in runners:
        print(f"[Long Benchmark] Running {runner.name} ...")
        try:
            summary = benchmark_long_runner(runner, samples)
            summaries.append(summary)
            for row in summary["samples"]:
                print(
                    f"  {row['sample_id']} ({row['duration_sec']}s): "
                    f"lat={row['latency_ms']}ms, rtf={row['rtf']}, "
                    f"wer={row['wer']*100:.1f}%, words={row['predicted_words']}/{row['expected_words']}"
                )
        except Exception as exc:
            print(f"  ! Skipped {runner.name}: {exc}")

    md_out = RESULTS_DIR / "long_audio_report.md"
    json_out = RESULTS_DIR / "long_audio_results.json"
    write_long_report(summaries, md_out, json_out)
    print(f"[Done] Saved long-audio reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
