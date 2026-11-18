"""Test 4: Meeting Vocabulary Hotword & Chunking Benchmark on Vietnamese-Accented Speech.

Evaluates how hotwords / initial_prompt (Whisper) and 4s chunking + post-STT phonetic
vocabulary snapping (Moonshine) improve technical term accuracy and stabilize RAM
compared to unguided baselines on L05_20s.wav and L01-L04.
"""

from __future__ import annotations

import argparse
from difflib import SequenceMatcher
import json
from pathlib import Path
import time
from typing import Any

from test_1_stt_benchmark import (
    MoonshineRunner,
    WhisperRunner,
    compute_wer,
    get_process_rss_mb,
    load_audio_mono_16k,
    normalize_text,
    phonetic_skeleton,
)
from test_3_long_audio_benchmark import LongSample, build_long_samples

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000

SAMPLE_HOTWORDS: dict[str, list[str]] = {
    "L01_6s": ["backend timeline", "database bottleneck"],
    "H01_10s_human": ["ever tried", "fail better"],
    "L02_11s": ["payment queue", "latency spikes", "deployment checklist", "stakeholders"],
    "L03_26s": ["sliding window", "WebRTC", "rollback strategy", "acceptance criteria"],
    "L04_42s": ["prefix pinning", "domain terminology", "speech recognition"],
    "L05_20s": [
        "respected instructor",
        "fellow classmates",
        "technical presentation",
        "ARP Poisoning",
        "MITM Architecture",
        "Cisco Enterprise Defense",
    ],
    "L06_76s": [
        "ARP",
        "Ethernet network",
        "Host A",
        "Gateway R1",
        "Layer 2",
        "MAC address",
        "ARP Request",
        "broadcast domain",
        "192.168.10.1",
        "Router R1",
        "ARP Reply",
        "ARP table",
        "Gratuitous ARP",
    ],
    "L07_33s": [
        "myself",
        "Sunday",
        "journey",
        "forgot",
        "write",
    ],
}


def build_prompt_string(terms: list[str]) -> str:
    return "Meeting vocabulary: " + ", ".join(terms) + "."


def compute_term_hit_rate(terms: list[str], predicted_text: str) -> tuple[float, list[str]]:
    """Measure how many target meeting terms appear in the normalized hypothesis."""
    if not terms:
        return 1.0, []
    norm_pred = normalize_text(predicted_text)
    matched: list[str] = []
    for term in terms:
        if normalize_text(term) in norm_pred:
            matched.append(term)
    return round(len(matched) / len(terms), 4), matched


def snap_vocabulary_ngrams(text: str, terms: list[str], threshold: float = 0.72) -> str:
    """Post-STT fuzzy/phonetic n-gram snapping against pinned meeting vocabulary."""
    words = text.split()
    if not words or not terms:
        return text

    term_entries = [(t, normalize_text(t).split()) for t in terms if normalize_text(t)]
    out: list[str] = []
    i = 0
    while i < len(words):
        replaced = False
        for raw_term, term_words in term_entries:
            n = len(term_words)
            if i + n > len(words):
                continue
            cand_slice = normalize_text(" ".join(words[i : i + n]))
            target_norm = " ".join(term_words)
            char_sim = SequenceMatcher(None, target_norm, cand_slice).ratio()
            skel_sim = SequenceMatcher(
                None,
                "".join(phonetic_skeleton(w) for w in term_words),
                "".join(phonetic_skeleton(w) for w in cand_slice.split()),
            ).ratio()
            if 0.6 * char_sim + 0.4 * skel_sim >= threshold:
                out.append(raw_term)
                i += n
                replaced = True
                break
        if not replaced:
            out.append(words[i])
            i += 1
    return " ".join(out)


class ChunkedMoonshineSnapRunner:
    """Moonshine runner slicing audio into <= chunk_sec windows + optional vocab snapping."""

    def __init__(
        self,
        model_name: str = "moonshine/base",
        precision: str = "quantized",
        chunk_sec: float = 4.5,
        snap_terms: list[str] | None = None,
    ) -> None:
        self._inner = MoonshineRunner(model_name=model_name, precision=precision)
        self.chunk_sec = chunk_sec
        self.snap_terms = snap_terms or []
        self.name = f"{model_name} ({precision}, chunk={chunk_sec}s+snap)"

    def load(self) -> None:
        self._inner.load()

    def transcribe(self, audio: Any, wav_path: Path | None = None) -> str:
        chunk_samples = int(self.chunk_sec * SAMPLE_RATE)
        parts: list[str] = []
        for start in range(0, len(audio), chunk_samples):
            segment = audio[start : start + chunk_samples]
            if len(segment) < int(0.2 * SAMPLE_RATE):
                continue
            parts.append(self._inner.transcribe(segment, wav_path))
        joined = " ".join(p.strip() for p in parts if p.strip()).strip()
        return snap_vocabulary_ngrams(joined, self.snap_terms) if self.snap_terms else joined

    def unload(self) -> None:
        self._inner.unload()


def run_hotword_eval(
    runner: Any,
    samples: list[LongSample],
    per_sample_hotwords: bool,
) -> dict[str, Any]:
    """Evaluate one runner across all samples, updating Whisper hotwords per sample if enabled."""
    rss_before = get_process_rss_mb()
    t_load = time.perf_counter()
    runner.load()
    load_ms = (time.perf_counter() - t_load) * 1000.0
    peak_rss = get_process_rss_mb()

    rows: list[dict[str, Any]] = []
    for sample in samples:
        if not sample.wav_path.exists():
            continue
        terms = SAMPLE_HOTWORDS.get(sample.sample_id, [])
        if per_sample_hotwords and isinstance(runner, WhisperRunner):
            runner.hotwords = ", ".join(terms)
            runner.initial_prompt = build_prompt_string(terms)
        elif per_sample_hotwords and isinstance(runner, ChunkedMoonshineSnapRunner):
            runner.snap_terms = terms

        audio = load_audio_mono_16k(sample.wav_path)
        t0 = time.perf_counter()
        pred = runner.transcribe(audio, sample.wav_path)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        peak_rss = max(peak_rss, get_process_rss_mb())

        hit_rate, matched_terms = compute_term_hit_rate(terms, pred)
        rows.append({
            "sample_id": sample.sample_id,
            "duration_sec": round(len(audio) / SAMPLE_RATE, 2),
            "latency_ms": round(lat_ms, 1),
            "wer": round(compute_wer(sample.expected_text, pred), 4),
            "term_hit_rate": hit_rate,
            "matched_terms": matched_terms,
            "target_terms": terms,
            "predicted_text": pred,
        })

    runner.unload()
    return {
        "engine": runner.name,
        "load_ms": round(load_ms, 1),
        "rss_delta_mb": round(max(0.0, peak_rss - rss_before), 1),
        "peak_rss_mb": round(peak_rss, 1),
        "samples": rows,
    }


def write_hotword_report(summaries: list[dict[str, Any]], md_path: Path, json_path: Path) -> None:
    lines = [
        "# Báo cáo Hotword & Vocabulary Biasing (Tập trung L05_20s giọng Việt)",
        "",
        "| Engine | RSS Delta (MB) | Sample | Latency (ms) | WER | Term Hit | Từ khóa khớp |",
        "| :--- | ---: | :--- | ---: | ---: | ---: | :--- |",
    ]
    for s in summaries:
        for r in s["samples"]:
            matched = ", ".join(r["matched_terms"]) if r["matched_terms"] else "(none)"
            lines.append(
                f"| {s['engine']} | {s['rss_delta_mb']} | {r['sample_id']} | "
                f"{r['latency_ms']} | {r['wer']*100:.1f}% | {r['term_hit_rate']*100:.1f}% | {matched} |"
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 4: Hotword Biasing Benchmark")
    parser.add_argument("--l05-only", action="store_true", help="Benchmark only L05_20s.wav for quick iteration")
    parser.add_argument("--sample", default=None, help="Optional sample_id filter (e.g. L06_76s)")
    parser.add_argument("--include-cuda", action="store_true", help="Also run Whisper base.en on CUDA FP16")
    args = parser.parse_args()

    samples = build_long_samples()
    if args.l05_only:
        samples = [s for s in samples if s.sample_id == "L05_20s"]
    elif args.sample:
        samples = [s for s in samples if s.sample_id == args.sample]

    configs: list[tuple[Any, bool]] = [
        (MoonshineRunner("moonshine/base", precision="quantized"), False),
        (ChunkedMoonshineSnapRunner("moonshine/base", precision="quantized", chunk_sec=4.5), True),
        (WhisperRunner("tiny.en", device="cpu", compute_type="int8", beam_size=1, vad_filter=False), False),
        (WhisperRunner("tiny.en", device="cpu", compute_type="int8", hotwords="active", beam_size=2, vad_filter=True), True),
        (WhisperRunner("base.en", device="cpu", compute_type="int8", hotwords="active", beam_size=2, vad_filter=True), True),
    ]
    if args.include_cuda:
        configs.append((WhisperRunner("base.en", device="cuda", compute_type="float16", hotwords="active", beam_size=2, vad_filter=True), True))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for runner, use_hw in configs:
        print(f"[Hotword Benchmark] Running {runner.name} (hotwords={use_hw}) ...")
        try:
            summary = run_hotword_eval(runner, samples, per_sample_hotwords=use_hw)
            summaries.append(summary)
            for row in summary["samples"]:
                print(
                    f"  {row['sample_id']}: lat={row['latency_ms']}ms, "
                    f"wer={row['wer']*100:.1f}%, term_hit={row['term_hit_rate']*100:.1f}% "
                    f"matched={row['matched_terms']}"
                )
        except Exception as exc:
            print(f"  ! Skipped {runner.name}: {exc}")

    md_out = RESULTS_DIR / "hotword_report.md"
    json_out = RESULTS_DIR / "hotword_results.json"
    write_hotword_report(summaries, md_out, json_out)
    print(f"[Done] Saved hotword reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
