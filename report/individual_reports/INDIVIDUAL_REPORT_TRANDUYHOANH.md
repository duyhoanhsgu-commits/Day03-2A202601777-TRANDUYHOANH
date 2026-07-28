# Báo Cáo Cá Nhân: Lab 3 — Chatbot vs ReAct Agent

- **Họ và tên**: Trần Duy Hoành
- **Mã học viên**: 2A202601777
- **Nhóm**: Bàn cuối dãy trái
- **Ngày nộp**: 28/07/2026

---

## I. Đóng Góp Kỹ Thuật (Technical Contribution - 15 Điểm)

### 1.1 Các Module Đã Trực Tiếp Triển Khai
1. **Mô-đun Công cụ (`src/tools/tools.py`)**:
   - Xây dựng 3 công cụ cốt lõi: `check_stock`, `get_discount`, `calc_shipping` cùng hệ thống giả lập cơ sở dữ liệu kho hàng, mã giảm giá và cước vận chuyển.
   - Chuẩn hóa đầu vào/đầu ra dưới dạng `dict` JSON (bọc exception thành dạng error envelope như `item_not_found`, `unknown_destination`, bảo đảm tool không bao giờ crash hệ thống).
2. **Mô-đun Chatbot Baseline (`src/chatbot/chatbot.py`)**:
   - Triển khai `EcommerceChatbot` chạy đúng **1 lần gọi LLM (1 LLM call, 0 tool calls)** với System Prompt đóng vai trợ lý bán hàng TechMart.
3. **Mô-đun ReAct Agent V1 & V2 (`src/agent/agent.py` & `src/agent/agent_v2.py`)**:
   - Viết bộ tách cú pháp `parse_action` (hỗ trợ positional args, kwargs, JSON format) và `parse_final_answer`.
   - Triển khai vòng lặp suy luận ReAct (`Thought -> Action -> Observation`) có giới hạn `max_steps` cứng và Safe Fallback.
   - Nâng cấp phiên bản `ReActAgentV2` bổ sung bộ phát hiện hành động lặp trùng (`repeated_action_detector`) và cơ chế gợi ý khôi phục lỗi (recovery hint).
4. **Mô-đun Đánh giá Định lượng (`scripts/run_lab_evaluation.py`)**:
   - Tạo kịch bản tự động chạy 5 test cases benchmark trên cả Chatbot và Agent, xuất dữ liệu thô ra [artifacts/evaluation/raw_results.json](file:///home/hoanh-tran/aithucchien/Day03-2A202601777-TRANDUYHOANH/artifacts/evaluation/raw_results.json).

### 1.2 Điểm Nổi Bật Về Mã Nguồn & 4 Invariants
Mã nguồn Agent tuân thủ nghiêm ngặt **4 Invariants** quan trọng:
- **I1 (Không lặp vô hạn)**: Giới hạn bởi `for step in range(1, max_steps + 1)`.
- **I2 (Một Action $\rightarrow$ Đúng một Observation)**: Mỗi nhánh lặp kết thúc bằng đúng 1 lần nối chuỗi `Observation:`.
- **I3 (Observation đưa vào Prompt bước sau)**: Nối trực tiếp kết quả tool thực tế vào ngữ cảnh trước khi gọi LLM ở bước kế tiếp.
- **I4 (Ứng dụng viết Observation, không phải Model)**: Loại bỏ triệt để trường hợp LLM tự bịa dòng `Observation:` bằng hàm cắt lọc chuỗi `content.split("Observation:")[0]`.

---

## II. Phân Tích Sự Cố & Chẩn Đoán Lỗi (Debugging Case Study - 10 Điểm)

### 2.1 Mô Tả Sự Cố (Problem Description)
Trong bài test giao dịch đa bước (mua 2 iPhone + mã WINNER + giao Hà Nội), phiên bản **ReAct Agent V1** rơi vào vòng lặp lặp lại hành động (`Repeated Action Livelock`). 
Agent gọi công cụ `check_stock("iPhone")` ở Bước 1, tiếp tục gọi lại chính công cụ và tham số này ở Bước 2 và Bước 3 mà không chuyển sang bước kiểm tra mã giảm giá hay tính phí ship.

### 2.2 Nguồn Nhật Ký & Dấu Vết (Log Trace)
- Tệp trace lỗi: [artifacts/traces/react_failure_trace.json](file:///home/hoanh-tran/aithucchien/Day03-2A202601777-TRANDUYHOANH/artifacts/traces/react_failure_trace.json)
- Log sự kiện: [logs/2026-07-28.log](file:///home/hoanh-tran/aithucchien/Day03-2A202601777-TRANDUYHOANH/logs)

### 2.3 Chẩn Đoán Nguyên Nhân Gốc Rễ (Root Cause Analysis)
- **Sai lệch đầu tiên (First Divergence)**: Bước 2.
- **Phân loại lỗi**: Lỗi vòng lặp (Loop Control Error).
- **Nguyên nhân cốt lõi**: Agent V1 thiếu bộ nhớ lưu vết các cặp `(tool_name, args)` đã gọi. Khi LLM phát lại Action cũ, Executor V1 vẫn chạy tool và trả về Observation y hệt. Do thông tin ngữ cảnh không thay đổi, LLM tiếp tục rơi vào trạng thái lặp vô hướng cho đến khi chạm mốc `max_steps`.

### 2.4 Giải Pháp Sửa Lỗi & Regression Test (Solution)
- **Nâng cấp Agent V2 (`src/agent/agent_v2.py`)**: Bổ sung tập hợp lưu các hành động đã thực thi `seen_actions`. Khi phát hiện hành động bị lặp lại:
  - V2 **không gọi lại tool** mà trả về thông báo cảnh báo lặp (`repeated_action_warning`) kèm gợi ý buộc LLM phải tổng hợp dữ liệu hoặc chuyển sang bước tiếp theo.
- **Regression Test**: Viết file [tests/test_agent_recovery.py](file:///home/hoanh-tran/aithucchien/Day03-2A202601777-TRANDUYHOANH/tests/test_agent_recovery.py) với test case `test_repeated_action_v1_fails` (**FAIL trên V1**) và `test_repeated_action_v2_recovers` (**PASS trên V2**).

---

## III. Bài Học & Góc Nhìn Cá Nhân: Chatbot vs ReAct (10 Điểm)

1. **Khả năng Suy luận (Reasoning)**:
   - Khối `Thought` đóng vai trò là "nháp suy nghĩ" (Chain of Thought). So với Chatbot trả lời ngay trong 1 bước, Agent dùng `Thought` để phân rã bài toán thành các bài toán nhỏ: *Cần hỏi giá $\rightarrow$ Cần hỏi giảm giá $\rightarrow$ Cần hỏi phí ship $\rightarrow$ Tổng hợp*.

2. **Độ Tin Cậy & Bằng Chứng (Groundedness & Reliability)**:
   - **Khi nào Chatbot thắng?** Đối với câu hỏi Q&A tĩnh (Chính sách đổi trả, Giờ làm việc), Chatbot cho phản hồi nhanh gấp **2.7 lần** và tiết kiệm token gấp **9.6 lần** so với Agent.
   - **Khi nào Agent thắng?** Đối với bài toán giao dịch đa bước, Chatbot hoàn toàn bịa số (Hallucination) do không có dữ liệu thực tế. Agent tuy đắt hơn nhưng cung cấp con số **100% kiểm chứng được (Grounded)** qua từng kết quả JSON của Tool.

3. **Phản hồi từ Môi trường (Observation Feedback)**:
   - `Observation` chính là điểm neo thực tế giúp Agent điều chỉnh hành vi. Ví dụ khi mã giảm giá trả về `valid: false`, Agent lập tức biết phải tính tổng tiền dựa trên giá gốc mà không bị hoảng hay crash hệ thống.

---

## IV. Đề Xuất Cải Tiến Cho Hệ Thống Production (5 Điểm)

1. **Khả năng Mở rộng (Scalability)**:
   - Khi hệ thống có hàng trăm công cụ, không thể nhét toàn bộ mô tả tool vào System Prompt. Cần áp dụng **Vector Search (RAG for Tools)** để chỉ chọn lọc top 3-5 tools phù hợp nhất với câu hỏi người dùng.
   - Hỗ trợ gọi tool bất đồng bộ/song song (`asyncio`) cho các công cụ độc lập (như kiểm tra kho và kiểm tra mã giảm giá).

2. **An toàn & Giám sát (Safety & Guardrails)**:
   - Bổ sung **Evidence Gate**: Buộc Agent chỉ được xuất ra `Final Answer` khi tất cả các con số trong câu trả lời đều có bằng chứng từ `Observation` tương ứng.
   - Xây dựng **Supervisor LLM** để kiểm duyệt các hành động có tác động thay đổi dữ liệu (như thanh toán, hủy đơn) trước khi thực thi.

3. **Tối ưu Chi phí (Cost Optimization)**:
   - Áp dụng cơ chế **Cache kết quả Tool**: Với các truy vấn trùng lặp trong thời gian ngắn (ví dụ giá iPhone), trả về kết quả từ Cache thay vì gọi lại database hoặc LLM.
