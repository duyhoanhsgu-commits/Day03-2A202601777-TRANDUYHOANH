# Group Report: Lab 3 — Production-Grade Agentic System

- **Team Name**: Bàn cuối dãy trái
- **Team Members**:
| STT | Họ và tên     | Mã học viên | Vai trò trong nhóm |
|-----|---------------|-------------|--------------------|
| 1   |Lê Quang Đức   |2A202601767  |Nhóm trưởng         |
| 2   |Đặng Trung Kiên|2A202601887  |Thành viên          |
| 3   |Trần Duy Hoành |2A202601777  |Thành viên          |

- **Deployment Date**: 2026-07-28

---

## 1. Executive Summary

Nhóm xây hai hệ thống trả lời **cùng một bộ 5 câu hỏi, cùng một model**: một chatbot baseline (một lần
gọi LLM, không tool) và một ReAct Agent (vòng lặp Thought–Action–Observation trên 4 tool).

- **Success Rate**: Chatbot **40%** (2/5) · Agent v2 **40%** (2/5) — trên `meta/llama-3.1-8b-instruct`
- **Grounded (câu trả lời có bằng chứng từ tool)**: Chatbot **0/5** · Agent v2 **4/5**
- **Hallucination**: cả hai **0%**
- **Key Outcome**: Success rate ngang nhau, nhưng hai hệ thống "thành công" theo hai cách khác hẳn.
  Chatbot đạt 40% bằng cách trả lời 2 câu không cần dữ liệu và **từ chối 60% còn lại**; nó chưa bao giờ
  *biết*, nó chỉ tránh sai. Agent trả lời được nhưng tốn **9,6 lần token** (3.092 vs 322) và **2,7 lần
  latency** (2.252ms vs 847ms). Cái nó mua được bằng chi phí đó không phải "đúng hơn" — mà là
  **kiểm chứng được**: mỗi con số truy ngược về một dòng JSON trong trace.

---

## 2. System Architecture & Tooling

### 2.1 ReAct Loop Implementation

Sơ đồ đầy đủ (5 diagram Mermaid, bám sát code): [`react_flowchart.md`](react_flowchart.md).

Vòng lặp trong `ReActAgent.run()` ép **4 invariant**, mỗi cái neo vào một dòng code cụ thể:

| # | Invariant | Thực thi ở đâu |
|---|---|---|
| I1 | Không lặp vô hạn | `for step in range(1, max_steps + 1)` |
| I2 | Một Action → đúng một Observation | mỗi nhánh loop kết thúc bằng đúng 1 lần nối `Observation:` |
| I3 | Observation vào prompt **trước** lần gọi kế tiếp | `transcript += ...` rồi mới `continue` |
| I4 | **Ứng dụng** viết Observation, không phải model | `content.split("Observation:")[0]` |

I4 đáng nói nhất: model rất hay tự viết luôn phần `Observation:` cho liền mạch văn. Đúng một dòng code
cắt bỏ nó — thiếu dòng đó thì agent tự nói chuyện với chính mình và mọi con số đều là bịa.
Test: `test_model_written_observations_are_discarded`.

Vòng lặp có **3 lối ra** (`final_answer`, `max_steps`, `provider_error`); V2 thêm `repeated_action`.

### 2.2 Tool Definitions (Inventory)

| Tool Name | Input | Output | Error trả về gợi ý gì |
|---|---|---|---|
| `check_stock` | `item_name: str` | `price`, `stock`, `weight_kg`, `status` | `item_not_found` → kèm `available_items` |
| `get_discount` | `coupon_code: str` | `discount_percent`, `valid`, `reason` | coupon sai/hết hạn **không phải lỗi** → `ok:true, valid:false` |
| `calc_shipping` | `weight: number`, `destination: str` | `shipping_cost`, `estimated_days` | `unknown_destination` → kèm `supported_destinations` |
| `calculate_total` | `unit_price`, `quantity`, `discount_percent`, `shipping_cost` | `subtotal`, `discount_amount`, `total`, `breakdown` | `invalid_argument` nếu `quantity < 1` hoặc không nguyên |

### Tool Design Evolution — vì sao có tool thứ tư

Ba tool đầu ra đời từ đề bài. Tool thứ tư ra đời từ **một lỗi quan sát được**.

Với 3 tool, agent lấy được giá **một chiếc** từ `check_stock`, rồi tự nhân với số lượng trong đầu.
Hệ quả: hỏi *"1 iPhone + WINNER + Hà Nội"* vẫn có nguy cơ nhận lại con số 45.038.000 của **2 chiếc** —
số lượng không hề là tham số của tool nào, nên không có gì ràng buộc nó, và không có bằng chứng nào để
đối chiếu.

`calculate_total` biến `quantity` thành **argument bắt buộc**: model phải đọc số lượng từ câu hỏi và
truyền vào, phép tính do Python làm, kết quả quay lại thành Observation kiểm chứng được. Kèm rule số 7
trong system prompt: *"Never do arithmetic yourself."*

Kiểm chứng bằng run thật (`artifacts/traces/success_v2_one_iphone.json`):

```
Action: calculate_total({"unit_price": 25000000, "quantity": 1, "discount_percent": 10, "shipping_cost": 38000})
Observation: {"ok": true, "quantity": 1, "subtotal": 25000000, "discount_amount": 2500000,
              "shipping_cost": 38000, "total": 22538000,
              "breakdown": "1 x 25,000,000 = 25,000,000; -10% = -2,500,000; + shipping 38,000 => 22,538,000 VND"}
```

Test `test_total_scales_with_quantity` chốt cả ba mốc: 1 chiếc → 22.538.000, 2 chiếc → 45.038.000,
3 chiếc → 67.538.000.

Bài học chung: **cái gì không đi qua tool thì không có bằng chứng.** Số lượng cũng là dữ liệu, và
để model tự xử lý trong đầu là bỏ ngỏ đúng chỗ dễ sai nhất.

Ba nguyên tắc thiết kế tool, mỗi cái có test riêng:

**1. Lỗi là dữ liệu, không phải exception.** Một `raise` sẽ giết vòng lặp ReAct. Tool nhận `None`, số
`123`, list rỗng — vẫn trả dict, không nổ (`test_no_tool_raises_on_garbage_input`).

**2. Lỗi phải kèm hint.** `check_stock('Samsung')` trả kèm danh sách `available_items`. Đây là khác biệt
giữa agent tự sửa được và agent đoán mò mãi.

**3. Câu trả lời nghiệp vụ không phải là lỗi.** MacBook hết hàng → `ok:true, status:"out_of_stock"`.
Mã `LEGACY` hết hạn → `ok:true, valid:false, discount 0%`. Nếu coi hai cái đó là lỗi, agent sẽ retry mãi
một mã không bao giờ hợp lệ (`test_expired_coupon_is_a_business_answer_not_an_error`).

Chuẩn hoá input vì LLM không gọn gàng: `"iphone"` = `"iPhone"`, `"Hà Nội"` = `"hanoi"` = `"hn"`,
`weight="0.8"` (string) = `0.8`. Riêng `bool` bị chặn tường minh vì `True` là subclass của `int` —
không chặn thì `calc_shipping(True, ...)` sẽ tính thành 1kg.

`TOOL_SPECS` (mô tả cho prompt) và `TOOL_REGISTRY` (hàm thật) được test là **khớp nhau**
(`test_registry_and_specs_describe_the_same_tools`) — một spec mô tả tool không tồn tại chính là cách
dạy agent hallucinate.

### 2.3 LLM Providers Used

Endpoint: **NVIDIA NIM** (`https://integrate.api.nvidia.com/v1`) qua `OpenAIProvider` — NIM tương thích
OpenAI nên tái dùng được nguyên SDK, chỉ đổi `base_url`.

- **Primary (đo chính thức)**: `meta/llama-3.1-8b-instruct`
- **Reference (model mạnh)**: `nvidia/nemotron-3-ultra-550b-a55b`
- **Provider abstraction**: `LLMProvider` (ABC) → `OpenAIProvider` / `GeminiProvider` / `LocalProvider`.
  `LocalProvider` import `llama_cpp` **lazy** để một dependency tuỳ chọn không làm sập cả test suite.

---

## 3. Telemetry & Performance Dashboard

Đo trên 5 case, `meta/llama-3.1-8b-instruct`, nguồn `artifacts/evaluation/raw_results.json`:

| Chỉ số | Chatbot | Agent v2 | Tỉ lệ |
|---|---:|---:|---|
| Avg latency | 847 ms | 2.252 ms | 2,7× |
| Avg tokens/task | 322 | 3.092 | 9,6× |
| Avg LLM calls | 1,0 | 4,6 | 4,6× |
| Avg tool calls | 0,0 | 2,4 | — |

**Cost**: `PerformanceTracker._calculate_cost()` hiện là **công thức giả** (`total_tokens/1000 × $0.01`),
chưa nối bảng giá thật của NVIDIA. Nhóm **không** đưa con số cost vào kết luận. Tỉ lệ token 9,6× là số
đo thật và đủ để nói về chi phí tương đối.

Mỗi bước ghi event JSON vào `logs/`: `AGENT_START`, `LLM_METRIC`, `TOOL_CALL`, `PARSE_ERROR`,
`REPEATED_ACTION`, `PROVIDER_ERROR`, `SAFE_FALLBACK`, `AGENT_END`.

---

## 4. Root Cause Analysis (RCA) — Failure Traces

Worksheet đầy đủ: [`RCA_repeated_action.md`](RCA_repeated_action.md).

### Case Study 1: Repeated Action Livelock

- **Input**: case 3 (2 iPhone + WINNER + Hà Nội, 0.8 kg)
- **Trace**: `artifacts/traces/failure_repeated_action_v1.json` — `meta/llama-3.1-8b-instruct`
- **Expected path**: `check_stock → get_discount → calc_shipping → Final Answer`
- **Actual path**: `check_stock → get_discount → calc_shipping → calc_shipping → calc_shipping → calc_shipping`
- **First divergence**: **bước 4**. Cả ba dữ kiện đã nằm trong transcript (25.000.000 / 10% / 38.000),
  đáng lẽ phải kết luận, nhưng model phát lại `calc_shipping` **y hệt**.
- **Error class**: **Loop** — không phải Parser (action parse sạch), không phải Tool (mọi observation `ok:true`).
- **Root cause**: V1 thực thi bất kỳ action nào nó nhận. Lời gọi trùng trả observation trùng → transcript
  không có thêm thông tin → trạng thái model ở bước sau về cơ bản không đổi. Livelock chỉ `max_steps` phá được.
- **Bằng chứng prompt không sửa được lỗi này**: ở bước 6 model tự viết *"I already have the shipping fee,
  I should not repeat the same action"* — rồi lặp lại. **Instruction-following không phải là cơ chế điều khiển.**
- **Smallest fix** (`src/agent/agent_v2.py`): nhớ mọi cặp `(tool, args đã sort)`. Gặp trùng thì **không chạy
  tool**, trả observation `repeated_action` kèm kết quả cache + lệnh kết luận. Lặp lần hai thì dừng.
  **Không** đụng parser, tool contract, câu chữ prompt hay `max_steps`.
- **Regression test**: `test_regression_fails_on_v1_passes_on_v2` — cùng một hàm assert chạy trên cả hai
  agent, `pytest.raises(AssertionError)` bọc V1, gọi trần trên V2.

**Before/after** (cùng script, cùng model, `max_steps=8`):

| Metric | V1 | V2 | Δ |
|---|---:|---:|---|
| termination | `max_steps` | `repeated_action` | dừng đúng nguyên nhân |
| llm_calls | 8 | 5 | −38% |
| tool_calls thực thi | 8 | 3 | −63% |
| lời gọi trùng bị chạy | 5 | **0** | triệt tiêu |
| tokens | 960 | 600 | −38% |

**Trade-off phải nói rõ**: V2 vẫn **chưa trả lời được** câu hỏi. Nó dừng sớm hơn, rẻ hơn, và báo cáo
bằng chứng đang có — nhưng phép tính cuối bỏ dở. Có test khẳng định V2 **không được** in ra 45.038.000
nếu nó chưa từng tính (`test_v2_fallback_reports_the_evidence_instead_of_a_generic_message`).

### Case Study 2: Hallucinated Argument

- **Input**: case 4 — *"Can I buy 1 MacBook and ship to Saigon?"*
- **Observation**: agent gọi `get_discount({"coupon_code": "SAIGON"})` — **lấy tên thành phố làm mã giảm giá**.
  Câu hỏi không hề nhắc tới coupon nào.
- **Root cause**: system prompt mô tả `get_discount` nhưng không nói rõ *khi nào KHÔNG dùng*. Model thấy
  có tool thì tìm cách dùng.
- **Vì sao không gây hại**: tool trả `ok:true, valid:false, reason:"not_found"` — thiết kế "câu trả lời
  nghiệp vụ không phải lỗi" khiến agent đọc và đi tiếp thay vì retry.
- **Fix đề xuất (chưa làm)**: thêm điều kiện "chỉ gọi `get_discount` khi khách hàng nêu mã cụ thể" vào
  description của tool. Đây là divergence khác → sẽ là V3.

### Case Study 3: Provider Outage

- **Sự cố thật**: một lượt gọi NVIDIA NIM treo **303 giây** (5 lượt còn lại chỉ ~1s), và
  `meta/llama-3.3-70b-instruct` timeout hoàn toàn.
- **Root cause**: SDK `openai` mặc định timeout **600s** — trong một agent loop, cái đó trông y hệt treo máy.
- **Fix**: `timeout=60s` + `max_retries=1` ở `OpenAIProvider`, và `_call_llm()` bọc try/except ở lớp
  `ReActAgent` (nên **cả V1 lẫn V2 dùng chung**). Lỗi provider → ghi trace, giữ nguyên transcript, retry sạch;
  quá `provider_error_budget` lần liên tiếp → dừng với `PROVIDER_ERROR_FALLBACK`.
- **Ràng buộc quan trọng**: mỗi lần lỗi vẫn **tiêu một step** — nếu không, provider hỏng liên tục sẽ vô
  hiệu hoá `max_steps` và phá vỡ invariant I1 (`test_a_provider_error_consumes_a_step_so_the_loop_stays_bounded`).

---

## 5. Ablation Studies & Experiments

### Experiment 1: Agent V1 vs V2 (thay đổi cơ chế, giữ nguyên model)

Xem bảng before/after ở mục 4. Kết luận: fix nằm ở **tầng orchestration**, không phải tầng prompt.

### Experiment 2: Chatbot vs Agent trên 5 case

Raw: `artifacts/evaluation/raw_results.json` · Lệnh: `python scripts/run_lab_evaluation.py --version v2`

| # | Case | Chatbot | Agent v2 | Winner |
|---|---|---|---|---|
| 1 | Return policy (static) | đúng, 1 call, 297 tok, **PASS** | gọi `check_stock`, 6 bước, 4.519 tok, **FAIL** | **Chatbot** |
| 2 | Working hours (static) | đúng, **PASS** | gọi `check_stock` ×2, **FAIL** | **Chatbot** |
| 3 | 2 iPhone + WINNER + Hà Nội | safe fallback, **FAIL** | `repeated_action`, **FAIL** | Hoà (cả hai fail) |
| 4 | MacBook hết hàng | safe fallback, **FAIL** | đúng path, **PASS** | **Agent** |
| 5 | iPad + LEGACY hết hạn | safe fallback, **FAIL** | đúng path, **PASS** | **Agent** |

Trên model tham chiếu `nemotron-3-ultra-550b-a55b`, **cùng file code**, case 3 xong trong 4 bước và ra
đúng **45.038.000 VND** (`artifacts/traces/success_v2_case3.json`) — trong khi 8b lặp vô hạn.
**Chất lượng agent phụ thuộc model nhiều hơn phụ thuộc code.**

### Experiment 3: Model selection benchmark

Đo 15 model bằng chính bài toán của lab, mỗi model 2 phép thử: (A) có phát ra `Action` parse được không,
(B) khi đã có đủ 3 observation thì có biết **dừng** và ra `Final Answer` đúng không.

| Model | A | B | Tổng đúng | lat A | lat B |
|---|---|---|---|---:|---:|
| `mistralai/mistral-nemotron` | OK | OK | OK | 0,9s | 4,2s |
| `nvidia/nemotron-3-super-120b-a12b` | OK | OK | OK | 4,5s | 2,3s |
| `nvidia/nemotron-3-ultra-550b-a55b` | OK | OK | OK | 1,3s | 3,4s |
| `meta/llama-3.1-70b-instruct` | OK | OK | OK | 42,7s | 4,7s |
| `meta/llama-3.1-8b-instruct` | OK | **LOOP** | sai | 1,4s | 4,4s |
| `openai/gpt-oss-120b` | OK | **NO_FINAL** | sai | 1,7s | 2,9s |

Hai phát hiện đáng giá:

1. Phép thử B **tái hiện lỗi lặp chỉ bằng 1 lượt gọi** thay vì cả một agent run 6 bước — công cụ chẩn
   đoán rẻ hơn 6 lần.
2. `nvidia/llama-3.3-nemotron-super-49b-v1.5` trả `content` **rỗng** vì nó là reasoning model, toàn bộ
   output nằm ở `reasoning_content`. To hơn ≠ hợp hơn.

---

## 6. Production Readiness Review

**Security**
- `.env` nằm trong `.gitignore`; `.env.example` chỉ chứa placeholder. Trong lab đã xảy ra một lần key thật
  lọt vào `.env.example` (file **được git theo dõi**) — phát hiện bằng `git log -S`, chưa vào commit nào.
- Tool nhận input từ LLM nên mọi tool đều validate kiểu và không bao giờ raise.
- Tool đều **read-only**; system prompt cấm tuyên bố đã đặt hàng. Agent chỉ báo giá, không mua.

**Guardrails** (đã có)
- `max_steps` chặn vòng lặp vô hạn.
- Repeated-action detector chặn livelock.
- `provider_error_budget` chặn treo do hạ tầng.
- I4 chặn model tự bịa Observation.

**Còn thiếu để lên production**
- Cost tracking thật (hiện là mock).
- Evidence gate: buộc phải có tool evidence trước khi cho `Final Answer`.
- Nhiều mặt hàng trong một đơn: `calculate_total` hiện chỉ nhận một `unit_price`/`quantity`.
- Retry có backoff; hiện retry ngay lập tức.

**Scaling**
- Nhiều tool → tool retrieval bằng vector search thay vì nhét hết vào prompt.
- Gọi tool song song khi các nhánh độc lập (`check_stock` và `get_discount` không phụ thuộc nhau).
- Supervisor LLM audit action trước khi thực thi, cho các tool có side effect.

---

## 7. Reproducibility

| Claim | Tái tạo bằng |
|---|---|
| Toàn bộ test | `python -m pytest -q` → **126 passed, 2 skipped** |
| Ground truth 45.038.000 (2 chiếc) | `tests/test_tools.py::test_lab_scenario_total_is_reproducible` |
| Số lượng được tôn trọng (1/2/3 chiếc) | `tests/test_tools.py::test_total_scales_with_quantity` |
| 1 iPhone → 22.538.000, live | `artifacts/traces/success_v2_one_iphone.json` |
| Tool contract & error envelope | `python scripts/demo_tools.py` |
| Bảng 5 case, mọi tỉ lệ | `python scripts/run_lab_evaluation.py --version v2` |
| V1 lặp 6 bước / 6 tool call | `artifacts/traces/failure_repeated_action_v1.json` |
| Nemotron 4 bước → 45.038.000 | `artifacts/traces/success_v2_case3.json` |
| Before/after V1 vs V2 | `python -m pytest tests/test_agent_recovery.py -q` |
| Regression FAIL-V1/PASS-V2 | `test_regression_fails_on_v1_passes_on_v2` |

**Test coverage**: 126 test, chạy không cần API key (chỉ 2 test SKIP vì `llama-cpp-python` là dependency
tuỳ chọn). Deterministic test đo **logic orchestration**; live run đo **hành vi model thật**. Hai loại
bằng chứng này được để riêng, không trộn.

---

## 8. Kết luận

Câu hỏi đúng không phải "agent hay chatbot tốt hơn", mà là: **câu trả lời này có cần bằng chứng không?**

- Không cần → agent chỉ là chatbot đắt gấp 10 lần. Case 1 và 2 chứng minh điều đó bằng số.
- Có cần → chatbot chỉ là cỗ máy bịa số rất trôi chảy, và `tool_calls == 0` là giới hạn **cấu trúc**,
  không phải giới hạn năng lực.

Thứ đắt nhất mà agent mua được không phải câu trả lời — mà là **cái trace**. Khi hệ thống sai, ta biết
sai ở bước nào. Chatbot sai thì chỉ biết là nó sai.
