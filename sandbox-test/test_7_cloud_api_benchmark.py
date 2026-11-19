"""Test 7: Free-Tier Cloud STT API Benchmark (Groq Whisper, Deepgram Nova-3 REST/WS, Gemini 2.5 Flash).

Evaluates cloud speech-to-text APIs on the exact same test set (L01-L07 + H01, especially
Vietnamese-accented L05_20s, L06_76s, L07_33s) to compare accuracy, term hit rate, and
end-to-end latency against local faster-whisper/base.en+hotwords and Sherpa-ONNX.

API keys are read strictly from environment variables (never hardcoded or saved to reports):
  - GROQ_API_KEY      -> Groq whisper-large-v3-turbo (REST + prompt biasing)
  - DEEPGRAM_API_KEY  -> Deepgram nova-3 (REST + WebSocket 200ms streaming + keyterm)
  - GEMINI_API_KEY    -> Google Gemini 2.5 Flash audio input (REST inline WAV)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode

import numpy as np
import requests
import soundfile as sf
import websockets

from test_1_stt_benchmark import compute_cer, compute_wer, load_audio_mono_16k
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


def load_dotenv_files() -> list[Path]:
    """Load KEY=VALUE pairs from sandbox-test/.env or repo-root .env into os.environ."""
    loaded: list[Path] = []
    for env_path in (SANDBOX_DIR / ".env", SANDBOX_DIR.parent / ".env"):
        if not env_path.is_file():
            continue
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip()
            if len(val) >= 2 and val[0] == val[-1] and val[0] in ('"', "'"):
                val = val[1:-1]
            if key and val and not os.environ.get(key):
                os.environ[key] = val
        loaded.append(env_path)
    return loaded


def audio_to_wav_bytes(audio_f32: np.ndarray) -> bytes:
    """Encode 16kHz mono float32 array as compact PCM16 WAV bytes in memory."""
    buf = io.BytesIO()
    sf.write(buf, audio_f32, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def audio_to_pcm16_bytes(audio_f32: np.ndarray) -> bytes:
    """Convert float32 [-1, 1] audio to raw little-endian PCM16 bytes."""
    clipped = np.clip(audio_f32, -1.0, 1.0)
    return (clipped * 32767.0).astype("<i2").tobytes()


def transcribe_groq(
    wav_bytes: bytes,
    terms: list[str],
    api_key: str,
    model: str = "whisper-large-v3-turbo",
) -> dict[str, Any]:
    """Call Groq OpenAI-compatible audio transcription endpoint with prompt biasing."""
    url = "https://api.groq.com/openai/v1/audio/transcriptions"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": ("sample.wav", wav_bytes, "audio/wav")}
    data = {
        "model": model,
        "language": "en",
        "prompt": build_prompt_string(terms),
        "response_format": "json",
        "temperature": "0",
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"Groq HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    return {
        "text": str(payload.get("text", "")).strip(),
        "latency_ms": lat_ms,
        "first_interim_ms": None,
        "finalize_ms": None,
    }


def transcribe_deepgram_rest(
    wav_bytes: bytes,
    terms: list[str],
    api_key: str,
    model: str = "nova-3",
) -> dict[str, Any]:
    """Call Deepgram REST pre-recorded endpoint with Nova-3 keyterm biasing."""
    params: list[tuple[str, str]] = [
        ("model", model),
        ("language", "en"),
        ("smart_format", "true"),
        ("punctuate", "true"),
    ]
    for term in terms:
        params.append(("keyterm", term))

    url = f"https://api.deepgram.com/v1/listen?{urlencode(params)}"
    headers = {
        "Authorization": f"Token {api_key}",
        "Content-Type": "audio/wav",
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, data=wav_bytes, timeout=60)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"Deepgram REST HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    alt = payload["results"]["channels"][0]["alternatives"][0]
    return {
        "text": str(alt.get("transcript", "")).strip(),
        "latency_ms": lat_ms,
        "first_interim_ms": None,
        "finalize_ms": None,
    }


async def _deepgram_ws_stream(
    audio_f32: np.ndarray,
    terms: list[str],
    api_key: str,
    model: str = "nova-3",
    chunk_ms: int = 200,
) -> dict[str, Any]:
    """Stream 200ms PCM16 chunks over Deepgram Nova-3 WebSocket and measure finalization."""
    params: list[tuple[str, str]] = [
        ("model", model),
        ("language", "en"),
        ("encoding", "linear16"),
        ("sample_rate", str(SAMPLE_RATE)),
        ("channels", "1"),
        ("interim_results", "true"),
        ("smart_format", "true"),
        ("punctuate", "true"),
    ]
    for term in terms:
        params.append(("keyterm", term))

    url = f"wss://api.deepgram.com/v1/listen?{urlencode(params)}"
    headers = {"Authorization": f"Token {api_key}"}

    pcm_bytes = audio_to_pcm16_bytes(audio_f32)
    bytes_per_chunk = max(320, int(SAMPLE_RATE * 2 * chunk_ms / 1000))

    final_parts: list[str] = []
    latest_interim = ""
    first_interim_ms: float | None = None
    t_stream_start = time.perf_counter()

    async with websockets.connect(url, additional_headers=headers, open_timeout=20) as ws:
        done_recv = asyncio.Event()

        async def receiver() -> None:
            nonlocal first_interim_ms, latest_interim
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if msg.get("type") == "Results":
                        alt = msg.get("channel", {}).get("alternatives", [{}])[0]
                        txt = str(alt.get("transcript", "")).strip()
                        if txt and first_interim_ms is None:
                            first_interim_ms = (time.perf_counter() - t_stream_start) * 1000.0
                        if msg.get("is_final") and txt:
                            final_parts.append(txt)
                            latest_interim = ""
                        elif txt:
                            latest_interim = txt
                    elif msg.get("type") == "Metadata":
                        done_recv.set()
            except websockets.ConnectionClosed:
                done_recv.set()

        recv_task = asyncio.create_task(receiver())
        for pos in range(0, len(pcm_bytes), bytes_per_chunk):
            await ws.send(pcm_bytes[pos : pos + bytes_per_chunk])
            await asyncio.sleep(0.02)

        t_finalize = time.perf_counter()
        await ws.send(json.dumps({"type": "Finalize"}))
        await ws.send(json.dumps({"type": "CloseStream"}))
        try:
            await asyncio.wait_for(recv_task, timeout=30.0)
        except asyncio.TimeoutError:
            recv_task.cancel()
        finalize_ms = (time.perf_counter() - t_finalize) * 1000.0

    total_ms = (time.perf_counter() - t_stream_start) * 1000.0
    transcript = " ".join(final_parts).strip() or latest_interim
    return {
        "text": transcript,
        "latency_ms": total_ms,
        "first_interim_ms": first_interim_ms,
        "finalize_ms": finalize_ms,
    }


def transcribe_deepgram_ws(
    audio_f32: np.ndarray,
    terms: list[str],
    api_key: str,
    model: str = "nova-3",
    chunk_ms: int = 200,
) -> dict[str, Any]:
    return asyncio.run(_deepgram_ws_stream(audio_f32, terms, api_key, model=model, chunk_ms=chunk_ms))


def resolve_gemini_models(api_key: str, preferred: str | None = None, run_all_flash: bool = False) -> list[str]:
    """Query ListModels to discover active Gemini generateContent models (avoiding deprecated 2.5)."""
    url = "https://generativelanguage.googleapis.com/v1beta/models"
    resp = requests.get(url, headers={"x-goog-api-key": api_key}, timeout=20)
    if resp.status_code >= 400:
        return [preferred or "gemini-3.5-flash-lite"]

    excluded_tokens = ("tts", "image", "robotics", "deep-research", "embedding", "live", "-2.5-", "-2.0-", "-1.5-", "-1.0-")
    available: list[str] = []
    for item in resp.json().get("models", []):
        methods = item.get("supportedGenerationMethods", [])
        name = str(item.get("name", "")).removeprefix("models/")
        low = name.lower()
        if "generateContent" in methods and "gemini" in low and not any(tok in low for tok in excluded_tokens):
            available.append(name)

    if available:
        print(f"[Test 7] Available active Gemini models: {', '.join(available[:12])}")

    if not available:
        return [preferred or "gemini-3.5-flash-lite"]
    if preferred and preferred in available:
        return [preferred]

    priority = [
        "gemini-3.8-flash",
        "gemini-3.5-flash-lite",
        "gemini-4-hydrogen",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3-flash",
        "gemini-3-flash-preview",
        "gemini-4-argon",
    ]
    matched_priority = [m for m in priority if m in available]
    if run_all_flash:
        fast = [m for m in available if any(k in m.lower() for k in ("flash", "hydrogen"))]
        ordered = matched_priority + [m for m in fast if m not in matched_priority]
        return ordered[:3] if ordered else available[:2]

    if matched_priority:
        return [matched_priority[0]]

    flash_models = [m for m in available if any(k in m.lower() for k in ("flash", "hydrogen"))]
    return [flash_models[0]] if flash_models else [available[0]]


def transcribe_gemini_transcribe(
    wav_bytes: bytes,
    terms: list[str],
    api_key: str,
    model: str = "gemini-3.5-transcribe",
) -> dict[str, Any]:
    """Call dedicated Gemini 3.5 Transcribe REST endpoint with customVocabulary biasing."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    gen_cfg: dict[str, Any] = {
        "languageCodes": ["en-US"],
        "temperature": 0.0,
    }
    if terms:
        gen_cfg["customVocabulary"] = terms[:1000]

    body = {
        "contents": [
            {
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                        }
                    }
                ]
            }
        ],
        "generationConfig": gen_cfg,
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini Transcribe HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    parts = payload["candidates"][0]["content"]["parts"]
    text = " ".join(str(p.get("text", "")).strip() for p in parts if "text" in p).strip()
    return {
        "text": text,
        "latency_ms": lat_ms,
        "first_interim_ms": None,
        "finalize_ms": None,
    }


async def _gemini_live_ws_stream(
    audio_f32: np.ndarray,
    terms: list[str],
    api_key: str,
    model: str = "gemini-3.5-transcribe-live",
    chunk_ms: int = 200,
) -> dict[str, Any]:
    """Stream 200ms PCM16 chunks over Gemini Live API BidiGenerateContent WebSocket."""
    base_uri = (
        "wss://generativelanguage.googleapis.com/ws/"
        "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
    )
    pcm_bytes = audio_to_pcm16_bytes(audio_f32)
    bytes_per_chunk = max(320, int(SAMPLE_RATE * 2 * chunk_ms / 1000))

    async def run_ws(uri: str, headers: dict[str, str] | None) -> dict[str, Any]:
        input_cfg: dict[str, Any] = {"mode": "smart"}
        if terms:
            input_cfg["custom_vocabulary"] = terms[:1000]

        setup_msg = {
            "setup": {
                "model": f"models/{model}",
                "generationConfig": {"responseModalities": ["TEXT"]},
                "inputAudioTranscription": input_cfg,
            }
        }

        chunks_text: list[str] = []
        interim_latest = ""
        first_interim_ms: float | None = None
        setup_done = asyncio.Event()
        turn_done = asyncio.Event()
        t_stream_start = time.perf_counter()

        async with websockets.connect(uri, additional_headers=headers, open_timeout=20) as ws:
            async def receiver() -> None:
                nonlocal first_interim_ms, interim_latest
                try:
                    async for raw in ws:
                        msg = json.loads(raw)
                        if "setupComplete" in msg:
                            setup_done.set()
                        sc = msg.get("serverContent")
                        if sc:
                            it = sc.get("inputTranscription")
                            if it and it.get("text"):
                                txt = str(it["text"]).strip()
                                if txt:
                                    if first_interim_ms is None:
                                        first_interim_ms = (time.perf_counter() - t_stream_start) * 1000.0
                                    chunks_text.append(txt)
                            iit = sc.get("interimInputTranscription")
                            if iit and iit.get("text"):
                                interim_latest = str(iit["text"]).strip()
                                if interim_latest and first_interim_ms is None:
                                    first_interim_ms = (time.perf_counter() - t_stream_start) * 1000.0
                            mt = sc.get("modelTurn")
                            if mt:
                                for part in mt.get("parts", []):
                                    if part.get("text"):
                                        txt = str(part["text"]).strip()
                                        if txt:
                                            if first_interim_ms is None:
                                                first_interim_ms = (time.perf_counter() - t_stream_start) * 1000.0
                                            chunks_text.append(txt)
                            if sc.get("turnComplete") or sc.get("generationComplete"):
                                turn_done.set()
                except websockets.ConnectionClosed:
                    setup_done.set()
                    turn_done.set()

            recv_task = asyncio.create_task(receiver())
            await ws.send(json.dumps(setup_msg))
            await asyncio.wait_for(setup_done.wait(), timeout=10.0)

            for pos in range(0, len(pcm_bytes), bytes_per_chunk):
                frame = pcm_bytes[pos : pos + bytes_per_chunk]
                msg = {
                    "realtimeInput": {
                        "mediaChunks": [
                            {
                                "mimeType": "audio/pcm;rate=16000",
                                "data": base64.b64encode(frame).decode("ascii"),
                            }
                        ]
                    }
                }
                await ws.send(json.dumps(msg))
                await asyncio.sleep(0.02)

            t_finalize = time.perf_counter()
            await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
            try:
                await asyncio.wait_for(turn_done.wait(), timeout=6.0)
            except asyncio.TimeoutError:
                pass
            finalize_ms = (time.perf_counter() - t_finalize) * 1000.0
            recv_task.cancel()

        total_ms = (time.perf_counter() - t_stream_start) * 1000.0
        final_text = " ".join(chunks_text).strip() or interim_latest
        return {
            "text": final_text,
            "latency_ms": total_ms,
            "first_interim_ms": first_interim_ms,
            "finalize_ms": finalize_ms,
        }

    try:
        return await run_ws(base_uri, {"x-goog-api-key": api_key})
    except Exception:
        return await run_ws(f"{base_uri}?key={api_key}", None)


def transcribe_gemini_live(
    audio_f32: np.ndarray,
    terms: list[str],
    api_key: str,
    model: str = "gemini-3.5-transcribe-live",
    chunk_ms: int = 200,
) -> dict[str, Any]:
    return asyncio.run(_gemini_live_ws_stream(audio_f32, terms, api_key, model=model, chunk_ms=chunk_ms))


def transcribe_gemini(
    wav_bytes: bytes,
    terms: list[str],
    api_key: str,
    model: str = "gemini-3.5-flash-lite",
) -> dict[str, Any]:
    """Call Google Gemini Flash generateContent with inline WAV audio for transcription."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }
    vocab_hint = ", ".join(terms) if terms else "general English conversation"
    prompt = (
        "Transcribe this English speech verbatim. Return ONLY the plain transcript text "
        "(no markdown, no explanation, no speaker labels). "
        f"Domain/meeting vocabulary that may appear: {vocab_hint}."
    )
    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": "audio/wav",
                            "data": base64.b64encode(wav_bytes).decode("ascii"),
                        }
                    },
                ]
            }
        ],
        "generationConfig": {"temperature": 0.0},
    }
    t0 = time.perf_counter()
    resp = requests.post(url, headers=headers, json=body, timeout=60)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    if resp.status_code >= 400:
        raise RuntimeError(f"Gemini HTTP {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    parts = payload["candidates"][0]["content"]["parts"]
    text = " ".join(str(p.get("text", "")).strip() for p in parts).strip()
    return {
        "text": text,
        "latency_ms": lat_ms,
        "first_interim_ms": None,
        "finalize_ms": None,
    }


def evaluate_provider(
    provider_name: str,
    samples: list[LongSample],
    fn: Any,
    inter_sample_delay_sec: float = 0.0,
) -> dict[str, Any]:
    """Run one cloud API provider across selected audio samples."""
    rows: list[dict[str, Any]] = []
    for idx, sample in enumerate(samples):
        if not sample.wav_path.exists():
            continue
        if idx > 0 and inter_sample_delay_sec > 0:
            time.sleep(inter_sample_delay_sec)

        terms = SAMPLE_HOTWORDS.get(sample.sample_id, [])
        audio = load_audio_mono_16k(sample.wav_path)
        wav_bytes = audio_to_wav_bytes(audio)

        out = fn(audio, wav_bytes, terms)
        pred = out["text"]
        hit_rate, matched_terms = compute_term_hit_rate(terms, pred)
        rows.append({
            "sample_id": sample.sample_id,
            "duration_sec": round(len(audio) / SAMPLE_RATE, 2),
            "latency_ms": round(out["latency_ms"], 1),
            "first_interim_ms": round(out["first_interim_ms"], 1) if out["first_interim_ms"] is not None else None,
            "finalize_ms": round(out["finalize_ms"], 1) if out["finalize_ms"] is not None else None,
            "wer": round(compute_wer(sample.expected_text, pred), 4),
            "cer": round(compute_cer(sample.expected_text, pred), 4),
            "term_hit_rate": hit_rate,
            "matched_terms": matched_terms,
            "predicted_text": pred,
        })

    return {"provider": provider_name, "samples": rows}


def write_cloud_report(summaries: list[dict[str, Any]], md_path: Path, json_path: Path) -> None:
    lines = [
        "# Báo cáo Kiểm thử Cloud STT API Free-Tier — Fluencieer Sandbox",
        "",
        "- **Mục tiêu:** So sánh độ sạch transcript (`WER`/`CER`), độ bắt thuật ngữ (`Term Hit`), và độ trễ của các API miễn phí/credit so với Local `faster-whisper/base.en+hotwords`.",
        "",
        "## 1. Bảng tổng hợp hiệu năng & độ chính xác",
        "",
        "| Provider | Sample | Duration (s) | Latency (ms) | First Interim (ms) | Finalize (ms) | WER | CER | Term Hit |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for s in summaries:
        for r in s["samples"]:
            fi = str(r["first_interim_ms"]) if r["first_interim_ms"] is not None else "-"
            fn = str(r["finalize_ms"]) if r["finalize_ms"] is not None else "-"
            lines.append(
                f"| {s['provider']} | {r['sample_id']} | {r['duration_sec']:.2f} | "
                f"{r['latency_ms']} | {fi} | {fn} | {r['wer']*100:.1f}% | "
                f"{r['cer']*100:.1f}% | {r['term_hit_rate']*100:.1f}% |"
            )

    lines.extend(["", "## 2. Transcript thực tế (So độ sạch hiển thị lên UI)", ""])
    for s in summaries:
        lines.append(f"### {s['provider']}")
        for r in s["samples"]:
            matched = ", ".join(r["matched_terms"]) if r["matched_terms"] else "(none)"
            lines.append(
                f"- **{r['sample_id']}** (Latency `{r['latency_ms']} ms`, WER `{r['wer']*100:.1f}%`, "
                f"Term Hit `{r['term_hit_rate']*100:.1f}%`, Khớp: `{matched}`): {r['predicted_text']}"
            )
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 7: Free-Tier Cloud STT API Benchmark")
    parser.add_argument("--vn-only", action="store_true", help="Benchmark only Vietnamese-accented L05-L07 samples")
    parser.add_argument("--gemini-only", action="store_true", help="Run only Gemini and merge into existing cloud_api_results.json")
    parser.add_argument("--gemini-all", action="store_true", help="Benchmark top active Gemini Flash/Hydrogen models returned by ListModels")
    parser.add_argument("--gemini-model", default=None, help="Preferred Gemini model ID for generateContent (default: auto-resolved via ListModels)")
    parser.add_argument("--sample", default=None, help="Filter to a single sample_id (e.g. L06_76s)")
    parser.add_argument("--groq-delay-sec", type=float, default=3.2, help="Pause between Groq requests to respect 20 RPM free tier")
    args = parser.parse_args()

    samples = build_long_samples()
    if args.vn_only:
        samples = [s for s in samples if s.sample_id in VN_SAMPLE_IDS]
    elif args.sample:
        samples = [s for s in samples if s.sample_id == args.sample]

    if not samples:
        raise SystemExit("No matching samples found.")

    loaded_envs = load_dotenv_files()
    if loaded_envs:
        print(f"[Test 7] Loaded env from: {', '.join(str(p) for p in loaded_envs)}")

    groq_key = os.environ.get("GROQ_API_KEY", "").strip()
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY", "").strip()
    gemini_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()

    jobs: list[tuple[str, Any, float]] = []
    if groq_key and not args.gemini_only:
        jobs.append((
            "groq/whisper-large-v3-turbo (REST+prompt)",
            lambda _a, wav, terms, k=groq_key: transcribe_groq(wav, terms, k),
            args.groq_delay_sec,
        ))
    if deepgram_key and not args.gemini_only:
        jobs.append((
            "deepgram/nova-3 (REST+keyterm)",
            lambda _a, wav, terms, k=deepgram_key: transcribe_deepgram_rest(wav, terms, k),
            0.0,
        ))
        jobs.append((
            "deepgram/nova-3 (WS 200ms stream+keyterm)",
            lambda audio, _w, terms, k=deepgram_key: transcribe_deepgram_ws(audio, terms, k),
            0.0,
        ))
    if gemini_key:
        if args.gemini_model:
            gem_models = [args.gemini_model.strip()]
        else:
            jobs.append((
                "google/gemini-3.5-transcribe (REST+customVocab)",
                lambda _a, wav, terms, k=gemini_key: transcribe_gemini_transcribe(wav, terms, k),
                1.5,
            ))
            jobs.append((
                "google/gemini-3.5-transcribe-live (WS 200ms stream+customVocab)",
                lambda audio, _w, terms, k=gemini_key: transcribe_gemini_live(audio, terms, k),
                1.0,
            ))
            gem_models = resolve_gemini_models(gemini_key, preferred=os.environ.get("GEMINI_MODEL", "").strip() or None, run_all_flash=args.gemini_all)

        print(f"[Test 7] Selected Gemini Flash/Hydrogen model(s): {', '.join(gem_models)}")
        for gem_model in gem_models:
            jobs.append((
                f"google/{gem_model} (REST inline WAV)",
                lambda _a, wav, terms, k=gemini_key, m=gem_model: transcribe_gemini(wav, terms, k, model=m),
                1.5,
            ))

    if not jobs:
        print("Chua tim thay API key nao trong bien moi truong.")
        print("Hay dat it nhat 1 key truoc khi chay (chi giu trong phien PowerShell):")
        print('  $env:GROQ_API_KEY = "gsk_..."')
        print('  $env:DEEPGRAM_API_KEY = "..."')
        print('  $env:GEMINI_API_KEY = "AIza..."')
        raise SystemExit(1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []

    for name, fn, delay in jobs:
        print(f"[Test 7] Running {name} ...")
        try:
            res = evaluate_provider(name, samples, fn, inter_sample_delay_sec=delay)
            summaries.append(res)
            for row in res["samples"]:
                print(
                    f"  {row['sample_id']}: lat={row['latency_ms']}ms "
                    f"wer={row['wer']*100:.1f}% term_hit={row['term_hit_rate']*100:.1f}%"
                )
        except Exception as exc:
            print(f"  ! Skipped {name}: {exc}")

    md_out = RESULTS_DIR / "cloud_api_report.md"
    json_out = RESULTS_DIR / "cloud_api_results.json"
    if args.gemini_only and json_out.exists():
        existing = json.loads(json_out.read_text(encoding="utf-8"))
        new_names = {s["provider"] for s in summaries}
        summaries = [s for s in existing if s.get("provider") not in new_names] + summaries
    write_cloud_report(summaries, md_out, json_out)
    print(f"[Done] Saved Cloud API benchmark reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
