"""Test 2: Local WebSocket (ws://localhost) IPC, Two-Tier VAD (200ms / 1.5-2s),
Pinned Prefix + Sliding Window (~100 MB), and <= 1.6 GB Memory Cap Verification.

Supports two modes:
  --mode suite   Run automated 5-case verification and save results/ws_vad_ram_report.md
  --mode server  Run persistent ws://127.0.0.1:8765 server for Chrome ws_browser_probe.html
"""

from __future__ import annotations

import argparse
import asyncio
from collections import deque
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import statistics
import time
from typing import Any

import numpy as np
import psutil
import websockets

SANDBOX_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SANDBOX_DIR / "results"

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
HOT_PAUSE_MS = 200
COOL_PAUSE_MS = 1600  # 1.5-2.0s turn-end / script reset window
SLIDING_WINDOW_CAP_MB = 100.0
TOTAL_RAM_CAP_MB = 1638.4  # 1.6 GB
IPC_TARGET_MS = 10.0


@dataclass
class TurnEntry:
    turn_id: int
    text: str
    kv_bytes: int


class ContextWindowManager:
    """Manage Pinned Prefix (meeting vocab) + Sliding Window (~100 MB) in cool phase."""

    def __init__(self, pinned_vocab: list[str], cap_mb: float = SLIDING_WINDOW_CAP_MB) -> None:
        self.pinned_vocab: list[str] = list(pinned_vocab)
        self.cap_bytes: int = int(cap_mb * 1024 * 1024)
        self.history: deque[TurnEntry] = deque()
        self.total_history_bytes: int = 0
        self.active_script: list[str] = []
        self.slide_runs: int = 0
        self.evicted_turns: int = 0

    def append_active_script(self, words: str) -> None:
        if words:
            self.active_script.append(words)

    def commit_and_slide_cool_phase(self, turn_id: int, simulated_kv_mb: float) -> dict[str, Any]:
        """Run during 1.5-2s cool pause: merge active script, reset UI script, slide history."""
        merged_text = " ".join(self.active_script).strip() or f"turn_{turn_id}"
        self.active_script.clear()

        kv_bytes = int(simulated_kv_mb * 1024 * 1024)
        self.history.append(TurnEntry(turn_id=turn_id, text=merged_text, kv_bytes=kv_bytes))
        self.total_history_bytes += kv_bytes
        self.slide_runs += 1

        while self.total_history_bytes > self.cap_bytes and len(self.history) > 1:
            removed = self.history.popleft()
            self.total_history_bytes -= removed.kv_bytes
            self.evicted_turns += 1

        return {
            "merged_text": merged_text,
            "pinned_vocab_count": len(self.pinned_vocab),
            "history_mb": round(self.total_history_bytes / (1024 * 1024), 2),
            "evicted_turns": self.evicted_turns,
        }

    def clear_session(self) -> float:
        """Clear all session dialogue data (FR-15) while preserving pinned vocab choice."""
        t0 = time.perf_counter()
        self.history.clear()
        self.active_script.clear()
        self.total_history_bytes = 0
        gc.collect()
        return (time.perf_counter() - t0) * 1000.0


class TwoTierVADState:
    """Detect 200ms hot trigger vs 1.5-2s cool turn-end trigger from 20ms PCM frames."""

    def __init__(self, rms_threshold: float = 0.015) -> None:
        self.rms_threshold = rms_threshold
        self.has_spoken_in_turn = False
        self.silence_ms = 0
        self.hot_fired_for_pause = False
        self.cool_fired_for_pause = False
        self.hot_triggers = 0
        self.cool_triggers = 0

    def process_frame(self, pcm_int16: bytes) -> list[str]:
        samples = np.frombuffer(pcm_int16, dtype=np.int16).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(samples * samples))) if len(samples) else 0.0
        events: list[str] = []

        if rms >= self.rms_threshold:
            self.has_spoken_in_turn = True
            self.silence_ms = 0
            self.hot_fired_for_pause = False
            self.cool_fired_for_pause = False
            return events

        if not self.has_spoken_in_turn:
            return events

        self.silence_ms += FRAME_MS
        if self.silence_ms >= HOT_PAUSE_MS and not self.hot_fired_for_pause:
            self.hot_fired_for_pause = True
            self.hot_triggers += 1
            events.append("hot_trigger_200ms")

        if self.silence_ms >= COOL_PAUSE_MS and not self.cool_fired_for_pause:
            self.cool_fired_for_pause = True
            self.cool_triggers += 1
            self.has_spoken_in_turn = False
            events.append("cool_reset_and_slide")

        return events


async def handle_ws_client(websocket: Any) -> None:
    """Handle binary PCM audio frames and JSON control events over ws://localhost."""
    vad = TwoTierVADState()
    ctx = ContextWindowManager(pinned_vocab=["scope", "bottleneck", "milestone", "latency"])
    turn_counter = 0

    async for message in websocket:
        if isinstance(message, bytes):
            events = vad.process_frame(message)
            for ev in events:
                if ev == "hot_trigger_200ms":
                    await websocket.send(
                        json.dumps({
                            "type": "hot_trigger_200ms",
                            "branches_ready": True,
                            "slide_runs": ctx.slide_runs,
                        })
                    )
                elif ev == "cool_reset_and_slide":
                    turn_counter += 1
                    slide_info = ctx.commit_and_slide_cool_phase(turn_counter, simulated_kv_mb=15.0)
                    await websocket.send(json.dumps({"type": "cool_reset_and_slide", **slide_info}))
            continue

        payload = json.loads(message)
        msg_type = payload.get("type")
        if msg_type == "ping":
            await websocket.send(json.dumps({"type": "pong", "client_ts": payload.get("client_ts")}))
        elif msg_type == "append_script":
            ctx.append_active_script(payload.get("text", ""))
        elif msg_type == "simulate_heavy_turn":
            turn_counter += 1
            slide_info = ctx.commit_and_slide_cool_phase(
                turn_counter,
                simulated_kv_mb=float(payload.get("kv_mb", 30.0)),
            )
            await websocket.send(json.dumps({"type": "slide_status", **slide_info}))
        elif msg_type == "close_session":
            clear_ms = ctx.clear_session()
            await websocket.send(
                json.dumps({
                    "type": "session_cleared",
                    "clear_ms": round(clear_ms, 2),
                    "history_len": len(ctx.history),
                })
            )


def make_voiced_frame() -> bytes:
    t = np.linspace(0, FRAME_MS / 1000.0, FRAME_SAMPLES, endpoint=False)
    wave = (0.25 * np.sin(2 * np.pi * 220 * t) * 32767.0).astype(np.int16)
    return wave.tobytes()


def make_silent_frame() -> bytes:
    return np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()


async def run_automated_suite(host: str, port: int) -> dict[str, Any]:
    """Run all 5 Fluencieer WebSocket / VAD / Sliding Window / RAM scenarios."""
    rss_start = psutil.Process().memory_info().rss / (1024 * 1024)
    server = await websockets.serve(handle_ws_client, host, port, max_size=2 * 1024 * 1024)
    uri = f"ws://{host}:{port}"

    voiced = make_voiced_frame()
    silent = make_silent_frame()
    case_results: list[dict[str, Any]] = []

    try:
        async with websockets.connect(uri) as ws:
            # Case 1: Measure ws://localhost IPC RTT (target <= 10 ms)
            rtts: list[float] = []
            for _ in range(50):
                t0 = time.perf_counter()
                await ws.send(json.dumps({"type": "ping", "client_ts": t0}))
                await ws.recv()
                rtts.append((time.perf_counter() - t0) * 1000.0)
            p95_rtt = float(np.percentile(rtts, 95))
            case_results.append({
                "case": "Case 1: ws://localhost IPC RTT",
                "target": f"p95 <= {IPC_TARGET_MS} ms",
                "observed": f"p50={statistics.median(rtts):.2f} ms, p95={p95_rtt:.2f} ms",
                "passed": p95_rtt <= IPC_TARGET_MS,
            })

            # Case 2: Hot trigger at 200ms pause (FR-3 / FR-7) without sliding window running
            await ws.send(json.dumps({"type": "append_script", "text": "We need to"}))
            for _ in range(15):  # 300ms voiced
                await ws.send(voiced)
            for _ in range(10):  # 200ms silence
                await ws.send(silent)
            hot_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            hot_ok = hot_msg["type"] == "hot_trigger_200ms" and hot_msg["slide_runs"] == 0
            case_results.append({
                "case": "Case 2: Hot trigger 200ms (FR-3/FR-7)",
                "target": "Trigger at 200ms; 0 slide runs in hot phase",
                "observed": f"event={hot_msg['type']}, slide_runs={hot_msg['slide_runs']}",
                "passed": hot_ok,
            })

            # Finish turn with 1.4s more silence -> reaches 1.6s cool phase (Case 4 prep)
            for _ in range(70):
                await ws.send(silent)
            cool_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))

            # Case 3: Fluent speech with 100ms micro-pauses (< 200ms) -> 0 false triggers (SM-C1)
            for _ in range(3):
                for _ in range(15):  # 300ms voiced
                    await ws.send(voiced)
                for _ in range(5):   # 100ms micro-pause (< 200ms)
                    await ws.send(silent)
            false_triggers = 0
            try:
                await asyncio.wait_for(ws.recv(), timeout=0.15)
                false_triggers += 1
            except asyncio.TimeoutError:
                pass
            case_results.append({
                "case": "Case 3: Fluent speech <200ms micro-pauses (SM-C1)",
                "target": "0 false triggers",
                "observed": f"false_triggers={false_triggers}",
                "passed": false_triggers == 0,
            })

            # Case 4: Sliding Window (~100 MB cap) & Pinned Prefix retention (FR-9 / Cách hiểu A)
            last_slide = cool_msg
            for _ in range(5):  # Push 5 x 30 MB = 150 MB (> 100 MB cap)
                await ws.send(json.dumps({"type": "simulate_heavy_turn", "kv_mb": 30.0}))
                last_slide = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            slide_ok = (
                last_slide["history_mb"] <= SLIDING_WINDOW_CAP_MB
                and last_slide["evicted_turns"] >= 1
                and last_slide["pinned_vocab_count"] == 4
            )
            case_results.append({
                "case": "Case 4: Pinned Prefix + Sliding Window <= 100 MB (FR-9)",
                "target": f"history_mb <= {SLIDING_WINDOW_CAP_MB} MB & pinned_vocab=4",
                "observed": (
                    f"history_mb={last_slide['history_mb']} MB, "
                    f"evicted={last_slide['evicted_turns']}, pinned={last_slide['pinned_vocab_count']}"
                ),
                "passed": slide_ok,
            })

            # Case 5: Memory cap <= 1.6 GB & < 1s session cleanup (FR-14 / FR-15)
            rss_peak = psutil.Process().memory_info().rss / (1024 * 1024)
            await ws.send(json.dumps({"type": "close_session"}))
            clear_msg = json.loads(await asyncio.wait_for(ws.recv(), timeout=1.0))
            ram_ok = rss_peak <= TOTAL_RAM_CAP_MB and clear_msg["clear_ms"] < 1000.0 and clear_msg["history_len"] == 0
            case_results.append({
                "case": "Case 5: Process RSS <= 1.6 GB & Cleanup < 1s (FR-14/FR-15)",
                "target": f"RSS <= {TOTAL_RAM_CAP_MB} MB & clear < 1000 ms",
                "observed": f"rss_start={rss_start:.1f} MB, rss_peak={rss_peak:.1f} MB, clear={clear_msg['clear_ms']} ms",
                "passed": ram_ok,
            })
    finally:
        server.close()
        await server.wait_closed()

    return {
        "ws_uri": uri,
        "cases": case_results,
        "all_passed": all(c["passed"] for c in case_results),
    }


def write_report(report: dict[str, Any], md_path: Path, json_path: Path) -> None:
    lines = [
        "# Báo cáo Kiểm chứng WebSocket, VAD 2 nhịp, Sliding Window & Trần RAM — Fluencieer",
        "",
        f"- **Endpoint:** `{report['ws_uri']}`",
        f"- **Kết quả chung:** `{'PASS' if report['all_passed'] else 'FAIL'}`",
        "",
        "| Kịch bản | Ngưỡng mục tiêu | Kết quả đo | Trạng thái |",
        "| :--- | :--- | :--- | :--- |",
    ]
    for c in report["cases"]:
        status = "PASS" if c["passed"] else "FAIL"
        lines.append(f"| {c['case']} | {c['target']} | {c['observed']} | {status} |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


async def run_persistent_server(host: str, port: int) -> None:
    print(f"[Server] Listening on ws://{host}:{port} for ws_browser_probe.html (Ctrl+C to stop)...")
    async with websockets.serve(handle_ws_client, host, port, max_size=2 * 1024 * 1024):
        await asyncio.Future()


def main() -> None:
    parser = argparse.ArgumentParser(description="Fluencieer Sandbox Test 2: WebSocket / VAD / Sliding Window / RAM")
    parser.add_argument("--mode", default="suite", choices=["suite", "server"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.mode == "server":
        try:
            asyncio.run(run_persistent_server(args.host, args.port))
        except KeyboardInterrupt:
            print("\n[Server] Stopped.")
        return

    report = asyncio.run(run_automated_suite(args.host, args.port))
    md_out = RESULTS_DIR / "ws_vad_ram_report.md"
    json_out = RESULTS_DIR / "ws_vad_ram_results.json"
    write_report(report, md_out, json_out)
    for c in report["cases"]:
        print(f"[{'PASS' if c['passed'] else 'FAIL'}] {c['case']}: {c['observed']}")
    print(f"[Done] Saved reports to {md_out} and {json_out}")


if __name__ == "__main__":
    main()
