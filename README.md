# Fluencieer 🎙️⚡

> **Real-Time Meeting Speaking Teleprompter & Anti-Hesitation Copilot for Global Tech Negotiations**  
> *Giải pháp trợ lý phát biểu và máy nhắc chữ thông minh nhúng thời gian thực cho Google Meet / Zoom — Triệt tiêu ngập ngừng, giải phóng phản xạ nói tiếng Anh chuyên nghiệp.*

---

## 🎯 1. Tổng quan & Triết lý thiết kế (Overview)

**Fluencieer** được thiết kế đặc thù cho kỹ sư, Tech Lead, Solution Architect và chuyên gia công nghệ tham gia các cuộc họp kỹ thuật, đàm phán hợp đồng hoặc bảo vệ phương án kiến trúc bằng tiếng Anh với đối tác quốc tế.

Hệ thống giải quyết dứt điểm nghịch lý: **Vốn từ vựng đọc hiểu (Passive Comprehension) rất tốt nhưng lại bị nghẽn truy xuất từ vựng chủ động (Active Speaking Block)** khi đối thoại trực tiếp dưới áp lực cao.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            FLUENCIEER FLOW                                  │
├─────────────────────┬───────────────────────────┬───────────────────────────┤
│ 1. PRE-MEETING      │ 2. IN-CALL REALTIME       │ 3. POST-MEETING           │
│ Nạp bối cảnh .docx  │ Prefix-Aligned Forking    │ Báo cáo đo lường ngắc ngứ │
│ Trích xuất glossary │ Bắt nhịp VAD ~200ms       │ Tự động sinh flashcard    │
│ Prefill KV Cache    │ Khóa nhánh Hands-free     │ Luyện phản xạ từ vựng     │
└─────────────────────┴───────────────────────────┴───────────────────────────┘
```

---

## ✨ 2. Tính năng đột phá (Core Features)

### 🌿 Cây phân nhánh thẳng cột tiền tố (Prefix-Aligned Forking Tree)
- Giữ trọn vẹn ngữ cảnh của câu đang nói ở dòng `[1]`, các dòng `[2]` và `[3]` được tự động thụt lề căn thẳng cột.
- Triệt tiêu 100% rác mắt, không lặp lại từ đã nói, giúp người dùng phóng tầm mắt đọc trơn một hơi mượt mà:
  ```text
  [1] We need to  ▸ optimize system architecture ▸ to minimize latency down to 50ms.
  [2]             ▸ review the deployment plan   ▸ with our backend engineering team.
  [3]             ▸ verify compliance with SLA   ▸ before scheduled production release.
  ```

### ⏱️ Đón đầu cuốn chiếu (Rolling Chunks & Anti-Hesitation)
- Thay vì bung cả đoạn văn bản dài gây ngợp mắt (Cognitive Overload), hệ thống chỉ phóng chiếu từng cụm 2–4 từ có nghĩa ngữ pháp hoàn chỉnh đón đầu nhịp thở.
- Khi người dùng dừng lại ~200ms, 3 ngã rẽ Top-K ngữ cảnh xuất hiện ngay tức thì (< 50ms) giúp duy trì dòng chảy lưu loát (Fluency-First).

### 🖥️ Trải nghiệm Zero Alt+Tab (In-Call Native HUD)
- Giao diện Side Panel / HUD nổi được tích hợp trực tiếp, đặt sát cạnh webcam giúp người nói duy trì eye-contact tự nhiên với đối tác.
- Loại bỏ hoàn toàn thói quen nguy hiểm: vừa họp vừa cuống cuồng bấm `Alt + Tab` tra từ trên Google Translate.

### 🎙️ Điều khiển rảnh tay & Acoustic Gating
- **Voice Snap-to-Branch**: Tự động nhận diện và khóa nhánh kịch bản ngay khi cất giọng đọc từ đầu tiên của nhánh đó.
- **Phím tắt phản xạ**: Hỗ trợ phím số `1`, `2`, `3` và phím `Space` (ngắt nhịp thủ công tức thì khi môi trường có tạp âm).
- **Acoustic Gating (Silent Ingestion)**: Thanh nhắc chữ hoàn toàn im lặng khi đối tác phát biểu; chỉ kích hoạt khi người dùng chuẩn bị mở lời.

### 📄 Nạp ngữ cảnh tài liệu dự án (Context Grounding)
- Kéo thả tệp tài liệu kỹ thuật (`.docx`) trước giờ họp để trích xuất thuật ngữ chuyên ngành trong < 2 giây.
- Tự động nạp vào bộ đệm ngữ cảnh (KV Cache Prefill), đảm bảo câu gợi ý luôn bám sát thông số kiến trúc và cam kết SLA thực tế của dự án.

---

## 🏛️ 3. Kiến trúc kỹ thuật (System Architecture)

```
┌───────────────────────────────────────────────────────────────┐
│                      FLUENCIEER PIPELINE                      │
└───────────────────────────────────────────────────────────────┘
   [Microphone] ──► Web Speech API (Client STT) ─┐
                                                 │
   [Meet CC] ────► DOM Mutation Observer ────────┼──► WebSocket (0x02 User / 0x01 Partner)
                                                 │       │
   [Tài liệu] ───► Docx Parser / Term Extractor ─┘       ▼
                                                  FastAPI Backend
                                                         │
                                  ┌──────────────────────┴──────────────────────┐
                                  ▼                                             ▼
                        Lookahead Cache (<0.5ms)                      Groq LLM Engine (<200ms)
                                  │                                  (llama-3.3-70b-versatile)
                                  └──────────────────────┬──────────────────────┘
                                                         ▼
                                            Prefix-Aligned Tree Output
```

- **Client:** Chrome Extension Manifest V3, Web Speech API, Web Audio API AnalyserNode.
- **Backend:** FastAPI (Python 3.11+), WebSockets song công, Pydantic v2.
- **AI Inference Engine:** Groq API (`llama-3.3-70b-versatile`), Grammar-Aware Fallback Engine.
- **Target SLA:** Lookahead Cache < 10ms, P99 Latency < 500ms.

---

## 📁 4. Cấu trúc thư mục (Repository Layout)

```text
Fluencieer/
├── backend/                       # FastAPI & WebSocket Backend Service
│   ├── app/
│   │   ├── config.py              # Cấu hình Pydantic & Biến môi trường Groq
│   │   ├── main.py                # FastAPI endpoints & WebSocket router
│   │   └── services/              # ASR, Chunk Predictor, Lookahead Cache, Docx Parser
│   ├── tests/                     # Backend Unit Tests
│   └── pyproject.toml             # Quản lý dependencies với uv
│
├── extension/                     # Chrome Extension (Manifest V3)
│   ├── manifest.json              # Khai báo Chrome Side Panel & Quyền
│   └── sidepanel/                 # Giao diện Cột Bên Cạnh (Side Panel)
│       ├── sidepanel.html         # Bố cục giao diện gọn gàng, trực quan
│       ├── sidepanel.css          # Dark Mode UI & Hiệu ứng phát sáng
│       ├── sidepanel.js           # Xử lý Web Speech, VAD, Voice Snap & WebSocket
│       └── permission.*           # Trang xin cấp quyền Microphone chuẩn CSP
│
├── tests/                         # Kịch bản kiểm thử tích hợp E2E
│   └── e2e/
│       ├── test_10step_trace.py   # Benchmark chuỗi 10 bước phát âm chuẩn
│       └── test_data/             # Tệp Word mẫu (.docx) phục vụ kiểm thử
│
├── docs/                          # Tài liệu kỹ thuật & Hướng dẫn triển khai
├── AGENTS.md                      # Quy tắc thực thi chuẩn mực cho Agents
└── README.md                      # Tài liệu tổng quan dự án
```

---

## 🚀 5. Khởi chạy nhanh (Quick Start)

### Yêu cầu hệ thống
- Python 3.11+ & công cụ quản lý gói [uv](https://github.com/astral-sh/uv)
- Google Chrome (phiên bản hỗ trợ Chrome Side Panel API)

### Bước 1: Khởi động Backend
```bash
# 1. Di chuyển vào thư mục backend
cd backend

# 2. Cài đặt dependencies
uv sync

# 3. Tạo file cấu hình môi trường (.env)
echo "GROQ_API_KEY=your_groq_api_key_here" > .env
echo "GROQ_MODEL=llama-3.3-70b-versatile" >> .env

# 4. Chạy server phát triển
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Bước 2: Cài đặt Chrome Extension
1. Mở Google Chrome và truy cập địa chỉ `chrome://extensions/`.
2. Bật chế độ **Developer mode** ở góc trên bên phải.
3. Bấm **Load unpacked** (Tải tiện ích đã giải nén) và chọn thư mục `extension/` của dự án.
4. Mở Side Panel trên trình duyệt, bấm **"Bật Mic nhận diện"** và bắt đầu trải nghiệm.

---

## 🧪 6. Kiểm thử tự động (Testing)

> *Toàn bộ test suite được thiết kế để kiểm tra độc lập và xác thực nghiêm ngặt độ trễ:*

```bash
cd backend

# Chạy Unit Tests
uv run pytest -v

# Chạy E2E Benchmark 10 bước phát âm & Kiểm tra SLA
uv run pytest ../tests/e2e/test_10step_trace.py -v -s
```

---

## 📄 7. Bản quyền & Giấy phép (License)

Dự án được phân phối dưới giấy phép [MIT License](LICENSE). Mọi đóng góp và cải tiến đều được hoan nghênh!
