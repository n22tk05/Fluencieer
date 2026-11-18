# Fluencieer — Sandbox Test Suite (`sandbox-test`)

Thư mục này chứa bộ kịch bản kiểm thử độc lập phục vụ **2 nút thắt Tuần 1** đã xác định trong [prd.md](file:///C:/Users/ADMIN/Desktop/TK/_bmad-output/planning-artifacts/prds/prd-fluencieer-2026-11-13/prd.md) và [addendum.md](file:///C:/Users/ADMIN/Desktop/TK/_bmad-output/planning-artifacts/prds/prd-fluencieer-2026-11-13/addendum.md):

1. **Test 1 (`test_1_stt_benchmark.py` — giải quyết `§9.2 #5`):** So sánh A/B thực tế giữa **Moonshine (`moonshine/tiny`, `moonshine/base`)** và **Faster-Whisper (`tiny.en`)** trên các câu nói ngắn 1–3 giây giọng Việt nói tiếng Anh (độ trễ p50/p95, RSS RAM / GPU VRAM, WER/CER và tỷ lệ khớp `FR-10` ở dung sai 70%–80%).
2. **Test 2 (`test_2_ws_audio_ram.py` + `ws_browser_probe.html` — giải quyết `§9.2 #1`, `FR-3`, `FR-9`, `FR-14`, `FR-15`):** Kiểm chứng cầu nối **`Web Audio API` → `ws://localhost` (~10 ms IPC)**, VAD 2 nhịp (**200 ms nóng** vs **1,5–2 giây nguội**), giữ cố định `Pinned Prefix` + trượt `Sliding Window (<= 100 MB)`, kiểm soát trần RAM `<= 1,6 GB` và xóa trắng phiên `< 1 giây`.

---

## 1. Cấu trúc Thư mục

- `requirements.txt` — Các thư viện Python cho cả 2 bộ test.
- `audio_samples/manifest.csv` — 24 kịch bản câu nói tiếng Anh ngắn 1–3 giây chia 4 nhóm (`prefix_200ms`, `voice_lock_keyword`, `offscript_phrase`, `domain_vocab`).
- `test_1_stt_benchmark.py` — Script benchmark A/B cho `Moonshine` vs `Faster-Whisper tiny.en`.
- `test_2_ws_audio_ram.py` — Script kiểm chứng `ws://localhost` IPC, VAD 2 nhịp, Sliding Window và trần RAM.
- `test_3_long_audio_benchmark.py` — Script test đoạn dài (6s–42s) đo độ trễ theo thời lượng, RTF, RAM tăng vọt của ONNX arena và WER.
- `ws_browser_probe.html` — Trang kiểm tra `Web Audio API` thu micro thật từ Chrome bắn qua `ws://127.0.0.1:8765`.
- `results/` — Thư mục tự sinh khi chạy test, lưu báo cáo `.md` và `.json`.

---

## 2. Cài đặt Môi trường (Chạy 1 lần trong `sandbox-test`)

```powershell
cd C:\Users\ADMIN\Desktop\TK\sandbox-test
uv venv .venv
.\.venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
```

---

## 3. Kế hoạch Kiểm thử & Lệnh chạy

### Kịch bản 1A: Kiểm chứng Pipeline Chấm điểm `FR-10` (Dry-run, chưa cần file ghi âm)
- **Mục tiêu:** Xác nhận bộ đọc `manifest.csv`, hàm tính WER/CER, hàm khớp âm vị/từ mờ (`70%–80%`) và xuất báo cáo hoạt động chính xác.
- **Kịch bản (`Input -> Expected Output`):** Dữ liệu giả lập 24 câu từ `manifest.csv` (mô phỏng nuốt âm cuối `/t, d, s, e/`) -> Xuất `results/stt_benchmark_report.md` và `results/stt_benchmark_results.json` với đầy đủ cột latency, WER và tỷ lệ khóa nhánh ở ngưỡng 70% & 80%.
- **Lệnh chạy:**
  ```powershell
  python test_1_stt_benchmark.py --dry-run
  ```

### Kịch bản 1B: Benchmark A/B Thực tế Giọng Việt (`Moonshine` vs `Whisper Tiny`)
- **Chuẩn bị đầu vào:** Ghi âm các file `.wav` (`16000 Hz`, `mono`, `16-bit PCM`) theo đúng tên file trong `audio_samples/manifest.csv` và lưu vào `sandbox-test/audio_samples/`.
- **Mục tiêu:** Đo xem giữa `moonshine/tiny`, `moonshine/base` và `whisper/tiny.en`, engine nào đạt độ trễ `< 100–150 ms` và tỷ lệ khớp `FR-10` cao nhất trên giọng Việt mà tổng RAM cộng với `Qwen2.5 1.5B` 4-bit vẫn nằm trong trần `<= 1,6 GB`.
- **Kịch bản (`Input -> Expected Output`):** Các file `.wav` 1–3s trong `audio_samples/` -> Nạp từng model 1 lần (warm inference), ghi nhận `p50`/`p95 latency`, `rss_delta_mb`, `mean_wer`, `voice_lock_hit_70` và `voice_lock_hit_80` vào `results/stt_benchmark_report.md`.
- **Lệnh chạy (CPU INT8 cho STT):**
  ```powershell
  python test_1_stt_benchmark.py --models moonshine/tiny moonshine/base whisper/tiny.en --device cpu --compute-type int8
  ```

### Kịch bản 2A: Kiểm chứng Tự động Cầu nối `ws://localhost`, VAD 2 Nhịp, Sliding Window & Trần RAM
- **Mục tiêu:** Xác minh 5 tiêu chí kiến trúc: (1) IPC `ws://localhost` p95 `<= 10 ms`; (2) Ngắt 200ms kích hoạt nóng không chạy trượt bộ nhớ; (3) Nghỉ `< 200 ms` không kích hoạt nhầm (`SM-C1 = 0`); (4) Nghỉ `1,6s` gộp lượt nói & cắt trượt `Sliding Window <= 100 MB` nhưng giữ nguyên `Pinned Prefix`; (5) RSS `<= 1638.4 MB (1.6 GB)` & xóa sạch phiên `< 1000 ms`.
- **Kịch bản (`Input -> Expected Output`):** Client Python gửi khung PCM 20ms (`16kHz`) & sự kiện JSON qua `ws://127.0.0.1:8765` -> Cả 5 case đạt `PASS` và xuất `results/ws_vad_ram_report.md`.
- **Lệnh chạy:**
  ```powershell
  python test_2_ws_audio_ram.py --mode suite
  ```

### Kịch bản 2B: Thử nghiệm Mic Thật bằng Chrome `Web Audio API`
- **Mục tiêu:** Kiểm chứng độ trễ RTT thực tế từ trình duyệt Chrome qua `ws://127.0.0.1:8765` và cảm nhận ngưỡng ngắt `200 ms` vs `1,6s`.
- **Kịch bản (`Input -> Expected Output`):** Bật server Python ở terminal 1, mở `ws_browser_probe.html` trên Chrome ở terminal 2, bấm **Bắt đầu Mic** và **Đo RTT (20 lần)** -> Log trình duyệt hiển thị `avg RTT <= 10 ms` và bắt đúng `hot_trigger_200ms` / `cool_reset_and_slide` khi ngừng nói.
- **Lệnh chạy:**
  ```powershell
  python test_2_ws_audio_ram.py --mode server
  ```
  *(Sau đó mở file `ws_browser_probe.html` trực tiếp bằng trình duyệt Chrome.)*
