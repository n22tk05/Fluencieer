# Fluencieer 🎙️⚡

> **Real-Time Meeting Speaking Teleprompter & Anti-Hesitation Copilot on Google Meet**  
> *Máy nhắc chữ phân nhánh thời gian thực nhúng thanh bên Google Meet, tự động bắt nhịp ngắt giọng tự nhiên để gợi ý 3 nhánh phát biểu tiếp theo căn thẳng cột; cho phép khóa nhánh bằng giọng nói hoàn toàn rảnh tay và xử lý 100% ngoại tuyến nhằm duy trì dòng chảy hội thoại tiếng Anh lưu loát trong mọi tình huống cuộc họp.*

---

## 🎯 1. Tầm nhìn & Định vị Sản phẩm (Vision & Positioning)

Trong môi trường làm việc toàn cầu, người đi làm và chuyên gia công nghệ thường có năng lực chuyên môn và vốn từ vựng đọc hiểu rất tốt, nhưng lại gặp điểm nghẽn **phản xạ truy xuất từ vựng chủ động (Active Vocabulary Retrieval Block)** khi đối thoại trực tiếp bằng tiếng Anh dưới áp lực thời gian. 

Chỉ một khoảng khựng 2–3 giây để tìm từ nối, thuật ngữ hay cách diễn đạt tự nhiên cũng dễ khiến người nói bị ngắt lời, phải chêm âm đệm kéo dài hoặc lén rời mắt khỏi camera để tra cứu từ điển, làm gián đoạn nhịp trao đổi và giảm tính thuyết phục.

**Fluencieer** được định vị là **trợ lý nhắc chữ phân nhánh thời gian thực** phục vụ **mọi cuộc họp trực tuyến trên Google Meet** (họp định kỳ nội bộ đa quốc gia, báo cáo tiến độ dự án, thảo luận giải pháp, phiên hỏi đáp bất ngờ cho đến đàm phán hợp tác), giải quyết dứt điểm các hạn chế của phần mềm nhắc chữ cuộn văn bản tĩnh hoặc các công cụ dịch thuật truyền thống:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             FLUENCIEER INTERACTION FLOW                          │
├───────────────────────┬────────────────────────────┬─────────────────────────────┤
│ 1. CHUẨN BỊ (PRE-MEET)│ 2. ĐỐI PHƯƠNG NÓI (LISTEN) │ 3. PHÁT BIỂU (SPEAK & SNAP) │
│ Nạp ghi chú / tài liệu│ Đọc phụ đề Meet (< 20ms)   │ Bắt nhịp ngắt giọng (200ms) │
│ Trích xuất từ vựng    │ Vùng gợi ý giữ tĩnh        │ Hiện 3 nhánh thẳng cột (<50ms)│
│ Prefill ngữ cảnh RAM  │ Giữ trọn sự tập trung      │ Khóa nhánh rảnh tay (<100ms)│
└───────────────────────┴────────────────────────────┴─────────────────────────────┘
```

---

## ✨ 2. Tính năng Cốt lõi (Core Capabilities)

### 🌿 Cây gợi ý tiền tố căn thẳng cột (Prefix-Aligned Tree)
- Giữ trọn vẹn cụm tiền tố người dùng vừa nói ở dòng `[1]`, các dòng `[2]` và `[3]` được tự động thụt lề căn thẳng hàng theo trục dọc từ vị trí từ tiếp theo.
- Triệt tiêu hoàn toàn rác mắt thị giác, không lặp lại từ đã nói, giúp người dùng lướt trọn 3 nhánh trong một ánh nhìn:
  ```text
  [1] We need to  ▸ clarify the project scope        ▸ before finalizing...
  [2]             ▸ update the implementation status  ▸ with all stakeholders...
  [3]             ▸ resolve the current bottleneck    ▸ to meet our deadline...
  ```

### ⏱️ Bắt nhịp ngắt giọng tự nhiên & Suy luận một lượt (< 50ms)
- Cảm biến âm thanh (VAD) phát hiện khoảng dừng lấy hơi hoặc ngập ngừng tự nhiên **200 miligiây** của người dùng để kích hoạt một lượt suy luận duy nhất (Single-pass top-K logits branching) trên mô hình cục bộ.
- Sinh song song 3 nhánh cụm từ (mỗi nhánh 3–4 token) kèm giải nghĩa ngắn gọn của từ khóa chính trong **dưới 50 miligiây**, hiển thị kịp thời trước khi kết thúc nhịp dừng lấy hơi tự nhiên.

### 🎙️ Khóa nhánh bằng giọng nói rảnh tay (Hands-Free Voice-Snap)
- **Voice Snap-to-Branch**: Người dùng chỉ cần đọc từ đầu tiên của nhánh đã chọn, hệ thống tự động khóa nhánh đó và ẩn 2 nhánh còn lại trong **dưới 100 miligiây** (với dung sai âm vị 70%–80% hỗ trợ cả người phát âm chưa chuẩn bản xứ hoặc nuốt âm đuôi).
- **Phím tắt dự phòng**: Trang bị phím số `1`, `2`, `3` để chọn nhánh tức thì khi môi trường văn phòng có tạp âm lớn.

### 💬 Phụ đề đối phương thời gian thực & Chống quá tải thị giác
- Tích hợp công cụ theo dõi DOM trích xuất phụ đề trực tiếp của Google Meet với độ trễ **dưới 20 miligiây**, hiển thị văn bản câu hỏi của đối phương ngay trên thanh bên giúp người dùng vừa nghe vừa đọc hiểu trọn vẹn nội dung.
- **Cognitive Guardrail**: Tuyệt đối giữ trống vùng gợi ý khi đối phương đang nói, chỉ kích hoạt khi người dùng bắt đầu mở lời để tránh gây mất tập trung.

### 🔄 Tự động làm mờ & Đồng bộ lại ngữ cảnh (Dynamic Re-anchoring)
- Khi người dùng nảy ra ý riêng và nói từ ngữ nằm ngoài 3 phương án gợi ý, giao diện tự động giảm độ đậm trong **dưới 50 miligiây** để không làm phân tâm nhịp nói.
- Ở lần ngắt giọng tiếp theo, hệ thống lấy chính từ vừa phát âm làm mốc ngữ cảnh mới để dựng ngay 3 nhánh đón đầu mới.

### 🕊️ Tôn trọng nhịp nói tự nhiên (Silent When Fluent)
- Khi người dùng đang phát biểu trôi chảy (các khoảng nghỉ giữa các từ dưới 200 miligiây), hệ thống hoàn toàn giữ im lặng, không cuộn chữ ép đọc và không hiển thị bất kỳ cảnh báo lỗi phát âm hay ngữ pháp nào.

### 📄 Nạp tài liệu & Định hướng ngữ cảnh cuộc họp (Context Grounding)
- Hỗ trợ nạp tệp tài liệu ghi chú (`.txt`, `.md`) hoặc dán văn bản chuẩn bị trước cuộc họp.
- Tự động trích xuất danh mục thuật ngữ trọng tâm nạp vào bộ nhớ để áp dụng bộ lọc xác suất (Trie-guided logit bias), định hướng mô hình ưu tiên gợi ý đúng chủ đề cuộc trao đổi.

---

## 🏛️ 3. Kiến trúc Kỹ thuật 100% Ngoại tuyến (Local Offline Architecture)

Fluencieer đặt cược vào kiến trúc **suy luận hoàn toàn trên thiết bị (On-Device Local Inference)** nhằm đáp ứng tiêu chuẩn khắt khe nhất về độ trễ thời gian thực và bảo mật thông tin cuộc họp doanh nghiệp:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       FLUENCIEER LOCAL PIPELINE                             │
└─────────────────────────────────────────────────────────────────────────────┘
  [Meet Live Captions] ──(DOM Observer < 20ms)──┐
                                                │
  [Microphone Audio]   ──(Local VAD ~200ms)─────┼──► [Shared Context Buffer]
                                                │          (RAM < 1.5 GB)
  [Tài liệu cuộc họp]  ──(Trie Bias Table)──────┘                 │
                                                                  ▼
                                                   ┌─────────────────────────┐
                                                   │ Qwen2.5 1.5B/1.7B 4-bit │
                                                   │   Single-Pass Softmax   │
                                                   └────────────┬────────────┘
                                                                ▼
                                                   Prefix-Aligned 3 Branches
                                                   (< 50ms display latency)
```

### Thông số Kỹ thuật & Ngân sách Độ trễ (Latency Budget)
| Thành phần / Mắt xích | Cơ chế kỹ thuật | Ngưỡng chỉ tiêu SLA |
| :--- | :--- | :--- |
| **Phụ đề đối phương** | DOM MutationObserver đọc Google Meet CC | `< 20 miligiây` |
| **Thu nhận âm thanh** | Web Audio API / VAD phát hiện khoảng ngắt giọng | `200 miligiây` |
| **Mô hình ngôn ngữ** | Qwen2.5 1.5B / 1.7B lượng tử hóa 4-bit (chạy nền cục bộ) | TTFT `20–40ms` (GPU) / `80–120ms` (CPU) |
| **Sinh nhánh gợi ý** | Trích xuất Top-K logits trực tiếp từ lớp phân phối softmax | `< 50 miligiây` |
| **Khóa nhánh giọng nói** | Khớp từ đầu nhánh với dung sai âm vị 70%–80% | `< 100 miligiây` |
| **Làm mờ khi nói lệch** | CSS opacity transition tự động ngắt hiển thị | `< 50 miligiây` |
| **Tổng trễ phản hồi** | Tính từ khi kết thúc khoảng ngắt giọng đến khi hiện chữ | `< 150 miligiây` (GPU) |
| **Trần bộ nhớ (RAM)** | Quản lý bộ đệm dùng chung (Shared KV-Cache prefix pinning) | `1.2 GB – 1.5 GB` |
| **Bảo mật & Riêng tư** | 100% Offline — xóa sạch bộ đệm ngữ cảnh khi đóng phiên | **Đúng 0 request ra Internet** |

---

## 🖥️ 4. Thiết kế Giao diện Thanh bên (Side Panel Wireframe)

Giao diện Fluencieer được thiết kế dưới dạng **Chrome Extension Side Panel** cố định cạnh cửa sổ Google Meet, chia đôi màn hình hợp lý giúp người nói duy trì hướng mắt nhìn gần camera:

```text
+-----------------------------------------------------------------+-----------------------------------+
|  GOOGLE MEET - Project Sync (meet.google.com/...)               |  FLUENCIEER SIDE PANEL (360px)    |
+-----------------------------------------------------------------+-----------------------------------+
|                                                                 | [A] Header & Trạng thái           |
|                  +-------------------------------+              |  ● Ngoại tuyến 100% | Local 4-bit |
|                  |                               |              |  RAM: 1.3 GB  | Mic: Bật (VAD)    |
|                  |     [ Video Đối Phương ]      |              +-----------------------------------+
|                  |                               |              | [B] Tài liệu & Từ vựng cuộc họp   |
|                  |    (Không bị thanh bên che,   |              |  📄 project-notes.md (Đã nạp)     |
|                  |    giữ mắt hướng về camera)   |              |  • Từ ưu tiên: scope, bottleneck  |
|                  +-------------------------------+              +-----------------------------------+
|                                                                 | [C] Phụ đề Đối phương (< 20ms)    |
|  [CC] Đối phương:                                               |  Sarah:                           |
|  "Could you share where we are with the backend timeline        |  "Could you share where we are    |
|   and whether any risks might affect next week's milestone?"    |   with the backend timeline...?"  |
|                                                                 +-----------------------------------+
|                                                                 | [D] Cây Gợi Ý Tiền Tố (< 50ms)    |
|                                                                 |  [1] We need to ▸ clarify the...  |
|                                                                 |  [2]            ▸ update the...   |
|                                                                 |  [3]            ▸ resolve the...  |
+-----------------------------------------------------------------+-----------------------------------+
|       [ Mic ]   [ Cam ]   [ CC: Bật ]   [ Rời cuộc họp ]        | [E] Đọc từ đầu nhánh | Phím 1 2 3 |
+-----------------------------------------------------------------+-----------------------------------+
```

> 💡 **Khám phá chi tiết Wireframe:**
> - Đặc tả trạng thái 7 giai đoạn: WIREFRAME.md
> - Bản mẫu HTML tương tác trực quan: wireframe.html

---

## 📄 5. Bản quyền & Giấy phép (License)

Dự án được bảo hộ và phân phối nội bộ. Mọi quyền sở hữu trí tuệ thuộc về nhóm phát triển Fluencieer.
