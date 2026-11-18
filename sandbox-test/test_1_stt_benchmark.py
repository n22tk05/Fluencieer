"""Test 1: A/B Benchmark for Local STT (Moonshine vs Faster-Whisper tiny.en).

Measures latency, memory footprint (RSS & optional GPU VRAM), WER/CER, and FR-10
Voice Lock keyword tolerance (70%-80% fuzzy/phonetic threshold) on short 1-3s
Vietnamese-accented English audio clips listed in audio_samples/manifest.csv.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from difflib import SequenceMatcher
import gc
import json
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import time
from typing import Any, Protocol

import numpy as np
import psutil

SANDBOX_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = SANDBOX_DIR / "audio_samples" / "manifest.csv"
AUDIO_DIR = SANDBOX_DIR / "audio_samples"
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000


class STTRunner(Protocol):
    """Unified interface for local STT benchmark engines."""

    name: str

    def load(self) -> None: ...
    def transcribe(self, audio: np.ndarray, wav_path: Path | None = None) -> str: ...
    def unload(self) -> None: ...


@dataclass
class SampleSpec:
    sample_id: str
    category: str
    wav_file: str
    expected_text: str
    target_keyword: str
    notes: str


@dataclass
class SampleResult:
    sample_id: str
    category: str
    duration_sec: float
    latency_ms: float
    expected_text: str
    predicted_text: str
    wer: float
    cer: float
    target_keyword: str
    keyword_score: float
    voice_lock_70: bool
    voice_lock_80: bool


def normalize_text(text: str) -> str:
    """Lowercase and strip punctuation while preserving apostrophes inside words."""
    cleaned = re.sub(r"[^a-z0-9'\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def levenshtein_distance(seq_a: list[str] | str, seq_b: list[str] | str) -> int:
    """Compute classic Levenshtein edit distance for sequences of chars or words."""
    if len(seq_a) < len(seq_b):
        seq_a, seq_b = seq_b, seq_a
    if not seq_b:
        return len(seq_a)
    prev = list(range(len(seq_b) + 1))
    for i, item_a in enumerate(seq_a):
        curr = [i + 1]
        for j, item_b in enumerate(seq_b):
            cost = 0 if item_a == item_b else 1
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    if not ref_words:
        return 0.0 if not hyp_words else 1.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def compute_cer(reference: str, hypothesis: str) -> float:
    ref_chars = normalize_text(reference)
    hyp_chars = normalize_text(hypothesis)
    if not ref_chars:
        return 0.0 if not hyp_chars else 1.0
    return levenshtein_distance(ref_chars, hyp_chars) / len(ref_chars)


def phonetic_skeleton(word: str) -> str:
    """Approximate Soundex-style skeleton resilient to VN-accent consonant drops."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return ""
    head = w[0]
    mapping = {
        "b": "1", "f": "1", "p": "1", "v": "1",
        "c": "2", "g": "2", "j": "2", "k": "2", "q": "2", "s": "2", "x": "2", "z": "2",
        "d": "3", "t": "3",
        "l": "4",
        "m": "5", "n": "5",
        "r": "6",
    }
    encoded: list[str] = []
    last_code = mapping.get(head, "")
    for ch in w[1:]:
        code = mapping.get(ch, "")
        if code and code != last_code:
            encoded.append(code)
        last_code = code
    return head + "".join(encoded)


def score_keyword_tolerance(target_keyword: str, predicted_text: str) -> float:
    """Score best fuzzy/phonetic match between target branch keyword and STT output."""
    target = normalize_text(target_keyword)
    if not target:
        return 0.0
    hyp_words = normalize_text(predicted_text).split()
    if not hyp_words:
        return 0.0
    target_skel = phonetic_skeleton(target)
    best = 0.0
    for word in hyp_words:
        char_sim = SequenceMatcher(None, target, word).ratio()
        skel_sim = SequenceMatcher(None, target_skel, phonetic_skeleton(word)).ratio()
        combined = 0.6 * char_sim + 0.4 * skel_sim
        if word == target:
            combined = 1.0
        best = max(best, combined)
    return round(best, 4)


def get_process_rss_mb() -> float:
    return psutil.Process().memory_info().rss / (1024 * 1024)


def get_gpu_vram_mb() -> float | None:
    """Return current NVIDIA GPU memory used in MiB if nvidia-smi is available."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            timeout=2,
        )
        first_line = out.strip().splitlines()[0]
        return float(first_line.strip())
    except Exception:
        return None


class MoonshineRunner:
    """Runner for useful-moonshine-onnx keeping model sessions resident in memory."""

    def __init__(self, model_name: str = "moonshine/tiny", precision: str = "quantized") -> None:
        self.model_name = model_name
        self.precision = precision
        self.name = f"{model_name} ({precision})"
        self._model: Any = None
        self._tokenizer: Any = None

    def load(self) -> None:
        from moonshine_onnx import MoonshineOnnxModel, load_tokenizer

        self._model = MoonshineOnnxModel(model_name=self.model_name, model_precision=self.precision)
        self._tokenizer = load_tokenizer()
        warmup = np.zeros((1, SAMPLE_RATE), dtype=np.float32)
        self._model.generate(warmup)

    def transcribe(self, audio: np.ndarray, wav_path: Path | None = None) -> str:
        batch = audio.astype(np.float32).reshape(1, -1)
        tokens = self._model.generate(batch)
        return self._tokenizer.decode(tokens[0])

    def unload(self) -> None:
        self._model = None
        self._tokenizer = None
        gc.collect()


class WhisperRunner:
    """Runner for faster-whisper models on CPU INT8 or CUDA FP16 with optional hotwords."""

    def __init__(
        self,
        model_size: str = "tiny.en",
        device: str = "cpu",
        compute_type: str = "int8",
        hotwords: str | None = None,
        initial_prompt: str | None = None,
        beam_size: int = 1,
        vad_filter: bool = False,
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.hotwords = hotwords
        self.initial_prompt = initial_prompt
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        hw_tag = "+hotwords" if (hotwords or initial_prompt) else ""
        self.name = f"faster-whisper/{model_size}{hw_tag} ({device}/{compute_type})"
        self._model: Any = None

    def load(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size,
            device=self.device,
            compute_type=self.compute_type,
        )
        warmup = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = self._model.transcribe(warmup, language="en", beam_size=self.beam_size)
        list(segments)

    def transcribe(self, audio: np.ndarray, wav_path: Path | None = None) -> str:
        segments, _ = self._model.transcribe(
            audio.astype(np.float32),
            language="en",
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            condition_on_previous_text=False,
            initial_prompt=self.initial_prompt,
            hotwords=self.hotwords,
        )
        return " ".join(seg.text.strip() for seg in segments).strip()

    def unload(self) -> None:
        self._model = None
        gc.collect()


class DryRunRunner:
    """Simulated STT runner to verify scoring, tolerance, and report pipeline offline."""

    def __init__(self, name: str, simulated_latency_ms: float, corrupt_tail: bool) -> None:
        self.name = name
        self._latency_ms = simulated_latency_ms
        self._corrupt_tail = corrupt_tail

    def load(self) -> None:
        return None

    def transcribe(self, audio: np.ndarray, wav_path: Path | None = None) -> str:
        time.sleep(self._latency_ms / 1000.0)
        return "__DRY_RUN__"

    def unload(self) -> None:
        return None


def load_manifest(path: Path) -> list[SampleSpec]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            SampleSpec(
                sample_id=row["id"].strip(),
                category=row["category"].strip(),
                wav_file=row["wav_file"].strip(),
                expected_text=row["expected_text"].strip(),
                target_keyword=row["target_keyword"].strip(),
                notes=row["notes"].strip(),
            )
            for row in reader
        ]


def load_audio_mono_16k(wav_path: Path) -> np.ndarray:
    """Load audio file, convert to mono, and downsample/resample to 16 kHz."""
    from math import gcd
    import soundfile as sf
    from scipy.signal import resample_poly

    data, sr = sf.read(wav_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != SAMPLE_RATE:
        factor = gcd(SAMPLE_RATE, sr)
        data = resample_poly(data, SAMPLE_RATE // factor, sr // factor)
    return data.astype(np.float32)


def simulate_vn_accent_prediction(expected: str, corrupt_tail: bool) -> str:
    """Produce deterministic simulated STT text for --dry-run pipeline validation."""
    words = expected.split()
    if not corrupt_tail:
        return expected
    altered = [w[:-1] if len(w) > 5 and w[-1] in "teds" else w for w in words]
    return " ".join(altered)


def evaluate_engine(
    runner: STTRunner,
    samples: list[SampleSpec],
    dry_run: bool,
) -> dict[str, Any]:
    """Benchmark a single STT engine across all available audio clips."""
    rss_before = get_process_rss_mb()
    vram_before = get_gpu_vram_mb()

    t_load_start = time.perf_counter()
    runner.load()
    load_ms = (time.perf_counter() - t_load_start) * 1000.0

    rss_after_load = get_process_rss_mb()
    peak_rss = rss_after_load
    vram_after_load = get_gpu_vram_mb()

    results: list[SampleResult] = []
    missing_files: list[str] = []

    for spec in samples:
        wav_path = AUDIO_DIR / spec.wav_file
        if dry_run:
            audio = np.zeros(int(SAMPLE_RATE * 1.5), dtype=np.float32)
            duration_sec = 1.5
        else:
            if not wav_path.exists():
                missing_files.append(spec.wav_file)
                continue
            audio = load_audio_mono_16k(wav_path)
            duration_sec = len(audio) / SAMPLE_RATE

        t0 = time.perf_counter()
        raw_pred = runner.transcribe(audio, wav_path)
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if raw_pred == "__DRY_RUN__":
            corrupt = getattr(runner, "_corrupt_tail", False)
            pred_text = simulate_vn_accent_prediction(spec.expected_text, corrupt)
        else:
            pred_text = raw_pred

        peak_rss = max(peak_rss, get_process_rss_mb())
        kw_score = score_keyword_tolerance(spec.target_keyword, pred_text)
        results.append(
            SampleResult(
                sample_id=spec.sample_id,
                category=spec.category,
                duration_sec=round(duration_sec, 2),
                latency_ms=round(latency_ms, 2),
                expected_text=spec.expected_text,
                predicted_text=pred_text,
                wer=round(compute_wer(spec.expected_text, pred_text), 4),
                cer=round(compute_cer(spec.expected_text, pred_text), 4),
                target_keyword=spec.target_keyword,
                keyword_score=kw_score,
                voice_lock_70=bool(spec.target_keyword and kw_score >= 0.70),
                voice_lock_80=bool(spec.target_keyword and kw_score >= 0.80),
            )
        )

    runner.unload()
    latencies = [r.latency_ms for r in results]
    kw_items = [r for r in results if r.target_keyword]

    summary = {
        "engine": runner.name,
        "samples_evaluated": len(results),
        "missing_files": missing_files,
        "load_ms": round(load_ms, 1),
        "rss_delta_mb": round(max(0.0, peak_rss - rss_before), 1),
        "peak_rss_mb": round(peak_rss, 1),
        "vram_delta_mb": (
            round(max(0.0, vram_after_load - vram_before), 1)
            if vram_before is not None and vram_after_load is not None
            else None
        ),
        "latency_p50_ms": round(statistics.median(latencies), 1) if latencies else None,
        "latency_p95_ms": (
            round(float(np.percentile(latencies, 95)), 1) if latencies else None
        ),
        "mean_wer": round(statistics.mean(r.wer for r in results), 4) if results else None,
        "mean_cer": round(statistics.mean(r.cer for r in results), 4) if results else None,
        "voice_lock_hit_70": (
            round(sum(1 for r in kw_items if r.voice_lock_70) / len(kw_items), 4)
            if kw_items
            else None
        ),
        "voice_lock_hit_80": (
            round(sum(1 for r in kw_items if r.voice_lock_80) / len(kw_items), 4)
            if kw_items
            else None
        ),
        "results": [asdict(r) for r in results],
    }
    return summary


def write_markdown_report(summaries: list[dict[str, Any]], out_path: Path, dry_run: bool) -> None:
    lines = [
        "# Báo cáo A/B Benchmark Local STT — Fluencieer Sandbox",
        "",
        f"- **Chế độ chạy:** `{'DRY-RUN (kiểm chứng pipeline)' if dry_run else 'LIVE AUDIO BENCHMARK'}`",
        "- **Mục tiêu đối chiếu PRD:** `FR-10` (< 100–150 ms Voice Lock, dung sai 70%–80%), `FR-12` (~150–200 ms ổn định), `FR-14` (tổng cụm <= 1,6 GB RAM).",
        "",
        "| Engine | Mẫu | RSS Delta (MB) | P50 Latency (ms) | P95 Latency (ms) | Mean WER | Lock >=70% | Lock >=80% |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        lock70 = f"{s['voice_lock_hit_70']*100:.1f}%" if s["voice_lock_hit_70"] is not None else "N/A"
        lock80 = f"{s['voice_lock_hit_80']*100:.1f}%" if s["voice_lock_hit_80"] is not None else "N/A"
        wer = f"{s['mean_wer']*100:.1f}%" if s["mean_wer"] is not None else "N/A"
        lines.append(
            f"| {s['engine']} | {s['samples_evaluated']} | {s['rss_delta_mb']} | "
            f"{s['latency_p50_ms']} | {s['latency_p95_ms']} | {wer} | {lock70} | {lock80} |"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_runners(args: argparse.Namespace) -> list[STTRunner]:
    if args.dry_run:
        return [
            DryRunRunner("dry-run/moonshine-tiny", simulated_latency_ms=45.0, corrupt_tail=True),
            DryRunRunner("dry-run/whisper-tiny.en", simulated_latency_ms=140.0, corrupt_tail=False),
        ]
    runners: list[STTRunner] = []
    for name in args.models:
        if name.startswith("moonshine/"):
            runners.append(MoonshineRunner(model_name=name, precision=args.moonshine_precision))
        elif name.startswith("whisper/"):
            size = name.split("/", 1)[1]
            runners.append(
                WhisperRunner(
                    model_size=size,
                    device=args.device,
                    compute_type=args.compute_type,
                    hotwords=args.hotwords,
                    initial_prompt=args.initial_prompt,
                    beam_size=2 if (args.hotwords or args.initial_prompt) else 1,
                )
            )
        else:
            raise ValueError(f"Unsupported model spec '{name}'. Use 'moonshine/tiny', 'moonshine/base', or 'whisper/tiny.en'.")
    return runners


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 1: Local STT A/B Benchmark")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["moonshine/tiny", "moonshine/base", "whisper/tiny.en"],
        help="Models to compare (e.g. moonshine/tiny moonshine/base whisper/tiny.en)",
    )
    parser.add_argument(
        "--moonshine-precision",
        default="quantized",
        choices=["float", "quantized", "quantized_4bit"],
        help="Precision for moonshine ONNX weights",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for faster-whisper")
    parser.add_argument("--compute-type", default="int8", help="CTranslate2 compute_type for faster-whisper")
    parser.add_argument("--hotwords", default=None, help="Comma-separated meeting vocabulary hotwords for Whisper")
    parser.add_argument("--initial-prompt", default=None, help="Initial vocabulary prompt for Whisper")
    parser.add_argument("--dry-run", action="store_true", help="Validate scoring & reporting without audio/models")
    args = parser.parse_args()

    samples = load_manifest(MANIFEST_PATH)
    runners = build_runners(args)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for runner in runners:
        print(f"[Benchmark] Evaluating {runner.name} ...")
        summary = evaluate_engine(runner, samples, dry_run=args.dry_run)
        summaries.append(summary)
        if summary["missing_files"]:
            print(f"  ! Missing {len(summary['missing_files'])} wav files in {AUDIO_DIR}")
        print(
            f"  -> evaluated={summary['samples_evaluated']}, p50={summary['latency_p50_ms']}ms, "
            f"p95={summary['latency_p95_ms']}ms, rss_delta={summary['rss_delta_mb']}MB"
        )

    json_out = RESULTS_DIR / "stt_benchmark_results.json"
    md_out = RESULTS_DIR / "stt_benchmark_report.md"
    json_out.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown_report(summaries, md_out, dry_run=args.dry_run)
    print(f"[Done] Saved reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
