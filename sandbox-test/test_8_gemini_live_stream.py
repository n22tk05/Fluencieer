"""Test 8: Real-Time Streaming Benchmark & Live Mic Bridge for gemini-3.5-transcribe-live.

Fixes mid-stream turnComplete early-close and duplicate transcript bugs observed in Test 7,
and provides two real-time streaming modes:
  1. File Stream Mode (default): Streams 200ms PCM16 frames at real-time pace (--pace 1.0)
     to verify interim latency, multi-turn continuity (L04, L06, L07), WER, and Term Hit.
  2. Live Browser Mic Bridge (--mic): Bridges Chrome Web Audio API (ws_browser_probe.html on
     ws://127.0.0.1:8765) directly to Gemini 3.5 Transcribe Live with real-time console & UI text.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
import websockets

from test_1_stt_benchmark import compute_cer, compute_wer, load_audio_mono_16k
from test_3_long_audio_benchmark import LongSample, build_long_samples
from test_4_hotword_benchmark import SAMPLE_HOTWORDS, compute_term_hit_rate
from test_7_cloud_api_benchmark import audio_to_pcm16_bytes, load_dotenv_files

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "results"
SAMPLE_RATE = 16000
VN_SAMPLE_IDS = {"L05_20s", "L06_76s", "L07_33s"}
GEMINI_LIVE_WS_URI = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)


async def _connect_gemini_live(api_key: str) -> Any:
    """Connect to Gemini Live BidiGenerateContent via header auth with query fallback."""
    try:
        return await websockets.connect(
            GEMINI_LIVE_WS_URI,
            additional_headers={"x-goog-api-key": api_key},
            open_timeout=20,
        )
    except Exception:
        return await websockets.connect(
            f"{GEMINI_LIVE_WS_URI}?key={api_key}",
            open_timeout=20,
        )


def build_live_setup(model: str, terms: list[str], mode: str = "smart") -> dict[str, Any]:
    input_cfg: dict[str, Any] = {"mode": mode}
    if terms:
        input_cfg["custom_vocabulary"] = terms[:1000]
    return {
        "setup": {
            "model": f"models/{model}",
            "generationConfig": {"responseModalities": ["TEXT"]},
            "inputAudioTranscription": input_cfg,
        }
    }


async def stream_audio_file_live(
    sample: LongSample,
    audio_f32: np.ndarray,
    terms: list[str],
    api_key: str,
    model: str = "gemini-3.5-transcribe-live",
    chunk_ms: int = 200,
    pace: float = 1.0,
    show_live: bool = True,
) -> dict[str, Any]:
    """Stream 200ms audio frames with proper multi-turn handling and live interim printout."""
    pcm_bytes = audio_to_pcm16_bytes(audio_f32)
    bytes_per_chunk = max(320, int(SAMPLE_RATE * 2 * chunk_ms / 1000))
    sleep_sec = (chunk_ms / 1000.0) * max(0.05, pace)

    final_input_parts: list[str] = []
    model_turn_fallback: list[str] = []
    latest_interim = ""
    first_interim_ms: float | None = None
    turn_boundaries = 0
    stream_ended = False

    setup_done = asyncio.Event()
    final_turn_done = asyncio.Event()
    last_event_ts = time.perf_counter()
    t_start = time.perf_counter()

    ws = await _connect_gemini_live(api_key)
    try:
        async def receiver() -> None:
            nonlocal first_interim_ms, latest_interim, turn_boundaries, last_event_ts
            try:
                async for raw in ws:
                    msg = json.loads(raw)
                    if "setupComplete" in msg:
                        setup_done.set()
                    sc = msg.get("serverContent")
                    if not sc:
                        continue

                    elapsed_s = time.perf_counter() - t_start
                    iit = sc.get("interimInputTranscription")
                    if iit and iit.get("text"):
                        txt = str(iit["text"]).strip()
                        if txt:
                            latest_interim = txt
                            last_event_ts = time.perf_counter()
                            if first_interim_ms is None:
                                first_interim_ms = elapsed_s * 1000.0
                            if show_live:
                                print(f"    [{elapsed_s:5.2f}s INTERIM] {txt}")

                    it = sc.get("inputTranscription")
                    if it and it.get("text"):
                        txt = str(it["text"]).strip()
                        if txt:
                            final_input_parts.append(txt)
                            latest_interim = ""
                            last_event_ts = time.perf_counter()
                            if first_interim_ms is None:
                                first_interim_ms = elapsed_s * 1000.0
                            if show_live:
                                print(f"    [{elapsed_s:5.2f}s FINAL  ] {txt}")

                    mt = sc.get("modelTurn")
                    if mt:
                        for part in mt.get("parts", []):
                            if part.get("text"):
                                txt = str(part["text"]).strip()
                                if txt:
                                    model_turn_fallback.append(txt)
                                    last_event_ts = time.perf_counter()

                    if sc.get("turnComplete") or sc.get("generationComplete"):
                        turn_boundaries += 1
                        if show_live:
                            print(f"    [{elapsed_s:5.2f}s TURN #{turn_boundaries}] (audio_ended={stream_ended})")
                        if stream_ended:
                            final_turn_done.set()
            except websockets.ConnectionClosed:
                setup_done.set()
                final_turn_done.set()

        recv_task = asyncio.create_task(receiver())
        await ws.send(json.dumps(build_live_setup(model=model, terms=terms)))
        await asyncio.wait_for(setup_done.wait(), timeout=10.0)

        for pos in range(0, len(pcm_bytes), bytes_per_chunk):
            frame = pcm_bytes[pos : pos + bytes_per_chunk]
            payload = {
                "realtimeInput": {
                    "mediaChunks": [
                        {
                            "mimeType": "audio/pcm;rate=16000",
                            "data": base64.b64encode(frame).decode("ascii"),
                        }
                    ]
                }
            }
            await ws.send(json.dumps(payload))
            await asyncio.sleep(sleep_sec)

        stream_ended = True
        final_turn_done.clear()
        t_flush = time.perf_counter()
        await ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))

        # Wait up to 12s for final turnComplete, or stop once quiet for 2.5s post-audioStreamEnd
        while not final_turn_done.is_set():
            now = time.perf_counter()
            if now - t_flush >= 12.0:
                break
            if now - t_flush >= 1.5 and now - last_event_ts >= 2.5:
                break
            try:
                await asyncio.wait_for(final_turn_done.wait(), timeout=0.3)
            except asyncio.TimeoutError:
                continue

        flush_ms = (time.perf_counter() - t_flush) * 1000.0
        recv_task.cancel()
    finally:
        await ws.close()

    total_ms = (time.perf_counter() - t_start) * 1000.0
    transcript = " ".join(final_input_parts).strip()
    if not transcript:
        transcript = " ".join(model_turn_fallback).strip() or latest_interim

    hit_rate, matched_terms = compute_term_hit_rate(terms, transcript)
    return {
        "sample_id": sample.sample_id,
        "duration_sec": round(len(audio_f32) / SAMPLE_RATE, 2),
        "pace": pace,
        "latency_ms": round(total_ms, 1),
        "first_interim_ms": round(first_interim_ms, 1) if first_interim_ms is not None else None,
        "flush_ms": round(flush_ms, 1),
        "turn_boundaries": turn_boundaries,
        "wer": round(compute_wer(sample.expected_text, transcript), 4),
        "cer": round(compute_cer(sample.expected_text, transcript), 4),
        "term_hit_rate": hit_rate,
        "matched_terms": matched_terms,
        "predicted_text": transcript,
    }


async def run_browser_mic_bridge(
    api_key: str,
    model: str,
    terms: list[str],
    host: str = "127.0.0.1",
    port: int = 8765,
) -> None:
    """Bridge Chrome ws_browser_probe.html (20ms PCM16 frames) to Gemini 3.5 Transcribe Live."""
    print(f"[Live Mic] Listening on ws://{host}:{port}")
    print("[Live Mic] Open sandbox-test/ws_browser_probe.html in Chrome and click 'Bat dau Mic'.")

    async def handle_browser(browser_ws: Any) -> None:
        print("[Live Mic] Browser connected. Opening Gemini Live session ...")
        gemini_ws = await _connect_gemini_live(api_key)
        setup_done = asyncio.Event()
        pcm_batch = bytearray()

        async def gemini_to_browser() -> None:
            try:
                async for raw in gemini_ws:
                    msg = json.loads(raw)
                    if "setupComplete" in msg:
                        setup_done.set()
                        continue
                    sc = msg.get("serverContent")
                    if not sc:
                        continue
                    iit = sc.get("interimInputTranscription")
                    if iit and iit.get("text"):
                        txt = str(iit["text"]).strip()
                        print(f"[INTERIM] {txt}")
                        await browser_ws.send(json.dumps({"type": "interim", "text": txt}))
                    it = sc.get("inputTranscription")
                    if it and it.get("text"):
                        txt = str(it["text"]).strip()
                        print(f"[FINAL  ] {txt}")
                        await browser_ws.send(json.dumps({"type": "final", "text": txt}))
            except websockets.ConnectionClosed:
                pass

        pump_task = asyncio.create_task(gemini_to_browser())
        await gemini_ws.send(json.dumps(build_live_setup(model=model, terms=terms)))
        await asyncio.wait_for(setup_done.wait(), timeout=10.0)
        await browser_ws.send(json.dumps({"type": "ready", "model": model}))

        try:
            async for message in browser_ws:
                if isinstance(message, bytes):
                    pcm_batch.extend(message)
                    # Batch 20ms browser frames into 160ms packets (5120 bytes @ 16kHz PCM16)
                    if len(pcm_batch) >= 5120:
                        chunk = bytes(pcm_batch)
                        pcm_batch.clear()
                        await gemini_ws.send(json.dumps({
                            "realtimeInput": {
                                "mediaChunks": [{
                                    "mimeType": "audio/pcm;rate=16000",
                                    "data": base64.b64encode(chunk).decode("ascii"),
                                }]
                            }
                        }))
                else:
                    ctrl = json.loads(message)
                    if ctrl.get("type") == "ping":
                        await browser_ws.send(json.dumps({"type": "pong", "client_ts": ctrl.get("client_ts")}))
                    elif ctrl.get("type") == "close_session":
                        if pcm_batch:
                            await gemini_ws.send(json.dumps({
                                "realtimeInput": {
                                    "mediaChunks": [{
                                        "mimeType": "audio/pcm;rate=16000",
                                        "data": base64.b64encode(bytes(pcm_batch)).decode("ascii"),
                                    }]
                                }
                            }))
                            pcm_batch.clear()
                        await gemini_ws.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))
                        await asyncio.sleep(1.5)
                        break
        finally:
            pump_task.cancel()
            await gemini_ws.close()
            print("[Live Mic] Session closed.")

    async with websockets.serve(handle_browser, host, port):
        await asyncio.Future()


def write_live_report(rows: list[dict[str, Any]], model: str, pace: float, md_path: Path, json_path: Path) -> None:
    lines = [
        f"# Báo cáo Real-Time Streaming Test — `{model}` (Pace `{pace}x`)",
        "",
        "- **Khắc phục so với Test 7:** Không đóng phiên tại `turnComplete` giữa file, tách riêng `inputTranscription` khỏi `modelTurn` để hết lặp câu, stream theo nhịp thực tế.",
        "",
        "## 1. Bảng kết quả Streaming",
        "",
        "| Sample | Duration (s) | Total (ms) | First Interim (ms) | Flush (ms) | Turns | WER | CER | Term Hit |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        fi = str(r["first_interim_ms"]) if r["first_interim_ms"] is not None else "-"
        lines.append(
            f"| {r['sample_id']} | {r['duration_sec']:.2f} | {r['latency_ms']} | {fi} | "
            f"{r['flush_ms']} | {r['turn_boundaries']} | {r['wer']*100:.1f}% | "
            f"{r['cer']*100:.1f}% | {r['term_hit_rate']*100:.1f}% |"
        )

    lines.extend(["", "## 2. Transcript hoàn chỉnh sau khi fix đóng lượt sớm", ""])
    for r in rows:
        matched = ", ".join(r["matched_terms"]) if r["matched_terms"] else "(none)"
        lines.append(
            f"- **{r['sample_id']}** (WER `{r['wer']*100:.1f}%`, Term Hit `{r['term_hit_rate']*100:.1f}%`, "
            f"Turns `{r['turn_boundaries']}`, Khớp: `{matched}`): {r['predicted_text']}"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Test 8: Gemini 3.5 Transcribe Live Real-Time Streamer")
    parser.add_argument("--mic", action="store_true", help="Run live Chrome Web Audio mic bridge on ws://127.0.0.1:8765")
    parser.add_argument("--hotwords", default="", help="Comma-separated hotwords for --mic mode")
    parser.add_argument("--vn-only", action="store_true", help="Stream only L05_20s, L06_76s, L07_33s")
    parser.add_argument("--sample", default=None, help="Stream single sample_id (e.g. L06_76s)")
    parser.add_argument("--pace", type=float, default=1.0, help="Streaming pace multiplier (1.0=real-time, 0.5=2x speed)")
    parser.add_argument("--quiet", action="store_true", help="Disable live line-by-line console printout")
    parser.add_argument("--model", default="gemini-3.5-transcribe-live", help="Gemini Live model ID")
    args = parser.parse_args()

    load_dotenv_files()
    api_key = (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing GEMINI_API_KEY in sandbox-test/.env or environment.")

    if args.mic:
        terms = [t.strip() for t in args.hotwords.split(",") if t.strip()]
        if not terms:
            terms = SAMPLE_HOTWORDS["L06_76s"] + SAMPLE_HOTWORDS["L05_20s"]
        asyncio.run(run_browser_mic_bridge(api_key=api_key, model=args.model, terms=terms))
        return

    samples = build_long_samples()
    if args.vn_only:
        samples = [s for s in samples if s.sample_id in VN_SAMPLE_IDS]
    elif args.sample:
        samples = [s for s in samples if s.sample_id == args.sample]
    if not samples:
        raise SystemExit("No matching samples found.")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for sample in samples:
        if not sample.wav_path.exists():
            continue
        terms = SAMPLE_HOTWORDS.get(sample.sample_id, [])
        audio = load_audio_mono_16k(sample.wav_path)
        print(f"\n[Test 8] Streaming {sample.sample_id} ({len(audio)/SAMPLE_RATE:.2f}s @ {args.pace}x) ...")
        res = asyncio.run(
            stream_audio_file_live(
                sample=sample,
                audio_f32=audio,
                terms=terms,
                api_key=api_key,
                model=args.model,
                pace=args.pace,
                show_live=not args.quiet,
            )
        )
        rows.append(res)
        print(
            f"  => RESULT {res['sample_id']}: wer={res['wer']*100:.1f}% "
            f"term_hit={res['term_hit_rate']*100:.1f}% turns={res['turn_boundaries']} "
            f"first_interim={res['first_interim_ms']}ms flush={res['flush_ms']}ms"
        )

    md_out = RESULTS_DIR / "gemini_live_stream_report.md"
    json_out = RESULTS_DIR / "gemini_live_stream_results.json"
    write_live_report(rows, model=args.model, pace=args.pace, md_path=md_out, json_path=json_out)
    print(f"\n[Done] Saved streaming report to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
