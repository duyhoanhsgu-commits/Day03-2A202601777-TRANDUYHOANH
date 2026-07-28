# ReAct Agent — Logic Flowchart

Sơ đồ bám sát `ReActAgent.run()` trong `src/agent/agent.py`. Bốn invariant được đánh dấu
**(I1)–(I4)** ngay tại vị trí code thực thi chúng.

| # | Invariant | Thực thi ở đâu |
| :-- | :--- | :--- |
| I1 | Không vòng lặp vô hạn — mọi run bị chặn bởi `max_steps` | `for step in range(1, max_steps + 1)` |
| I2 | Đúng **một** Observation cho mỗi Action | mỗi nhánh trong loop đều kết thúc bằng đúng 1 lần nối `Observation:` |
| I3 | Observation được nối vào prompt **trước** lần gọi LLM kế tiếp | `transcript += ...` rồi mới `continue` |
| I4 | **Ứng dụng** viết Observation, không phải model | `content.split("Observation:")[0]` |

> Sơ đồ 1–2 và 5 mô tả `ReActAgent` (V1). Sơ đồ 3 là điểm khác biệt duy nhất của `ReActAgentV2`.
> Guard lỗi provider nằm ở `_call_llm()` của lớp cha nên **cả hai version dùng chung**.

---

## 1. Vòng lặp chính — Thought / Action / Observation

```mermaid
flowchart TD
    START(["run(user_input)"]) --> INIT["log AGENT_START<br/>transcript = 'Question: ...'<br/>llm_calls = tool_calls = 0<br/>termination = 'max_steps'"]
    INIT --> GATE{"step &le; max_steps ?<br/><b>(I1)</b>"}

    GATE -- "No" --> FALLBACK["final_answer = SAFE_FALLBACK<br/>log SAFE_FALLBACK"]

    GATE -- "Yes" --> LLM["_call_llm(transcript, system_prompt)<br/>bọc try/except — không bao giờ raise"]
    LLM -- "provider_error<br/>(timeout / 5xx)" --> PERR{"quá provider_error_budget ?"}
    PERR -- "No" --> RETRY["log PROVIDER_ERROR<br/>transcript GIỮ NGUYÊN &rarr; retry sạch"]
    RETRY --> NEXT
    PERR -- "Yes" --> PFALL["termination = 'provider_error'<br/>PROVIDER_ERROR_FALLBACK"]
    PFALL --> RET

    LLM -- "OK" --> STRIP["raw = content.split('Observation:')[0]<br/>llm_calls += 1<br/><b>(I4)</b> bỏ mọi thứ model tự bịa sau 'Observation:'"]
    STRIP --> THOUGHT["parse_thought(raw) &rarr; ghi vào trace"]
    THOUGHT --> PARSE{"parse_action(raw)"}

    PARSE -- "ActionParseError<br/>(args không phải JSON hợp lệ)" --> OBS_PARSE["observation = <br/>{ok: false, error: 'parse_error'}<br/>log PARSE_ERROR"]
    PARSE -- "(tool_name, args)" --> EXEC["_execute_tool(tool_name, args)<br/>tool_calls += 1<br/>log TOOL_CALL"]
    PARSE -- "None (không có Action)" --> FINAL{"parse_final_answer(raw)"}

    FINAL -- "có Final Answer" --> DONE["final_answer = answer<br/>termination = 'final_answer'"]
    FINAL -- "None" --> OBS_NOACT["observation = <br/>{ok: false, error: 'no_action'}<br/>'Emit an Action or a Final Answer.'"]

    EXEC --> OBS_TOOL["observation = kết quả tool<br/><b>(I2)</b> 1 Action &rarr; đúng 1 Observation"]

    OBS_PARSE --> APPEND
    OBS_TOOL --> APPEND
    OBS_NOACT --> APPEND

    APPEND["transcript += raw + 'Observation: ' + json(observation)<br/><b>(I3)</b> nối TRƯỚC lần gọi LLM kế tiếp<br/>trace.append(entry)"]
    APPEND --> NEXT["step += 1"]
    NEXT --> GATE

    DONE --> RET
    FALLBACK --> RET

    RET["log AGENT_END<br/>return {content, llm_calls, tool_calls,<br/>steps, termination, latency_ms, usage, trace}"]
    RET --> END([" "])

    classDef guard fill:#fff3cd,stroke:#b8860b,color:#000
    classDef err fill:#f8d7da,stroke:#c0392b,color:#000
    classDef ok fill:#d4edda,stroke:#2e7d32,color:#000
    class STRIP,APPEND,GATE,PERR guard
    class OBS_PARSE,OBS_NOACT,FALLBACK,PFALL err
    class DONE,OBS_TOOL ok
```

**Điểm mấu chốt:** cả ba nhánh lỗi *của model* (`parse_error`, `no_action`, tool trả `ok:false`) đều
**không thoát vòng lặp** — chúng biến thành Observation có cấu trúc và quay lại cho model tự sửa.
Chỉ lỗi *hạ tầng* (provider chết liên tiếp) mới cắt ngang.

Vòng lặp có đúng **ba** lối ra: `final_answer`, `max_steps`, `provider_error`. V2 thêm lối ra thứ tư
là `repeated_action` (sơ đồ 3).

Lưu ý về lỗi provider: transcript **không** bị thay đổi khi retry — bước sau gửi lại đúng chuỗi cũ,
nên observation đã thu được không mất. Nhưng mỗi lần lỗi vẫn **tiêu một step**, nếu không thì một
provider hỏng liên tục sẽ vô hiệu hoá `max_steps` và phá vỡ invariant I1.

---

## 2. `_execute_tool()` — mọi lỗi đều là DATA, không bao giờ raise

```mermaid
flowchart TD
    IN(["_execute_tool(tool_name, args)"]) --> LOOKUP{"registry.get(tool_name)"}

    LOOKUP -- "None" --> UNKNOWN["{ok: false, error: 'unknown_tool',<br/>available_tools: [...]}"]
    LOOKUP -- "found" --> CALL["tool(**args)"]

    CALL -- "TypeError" --> BADARGS["{ok: false, error: 'bad_arguments',<br/>received: args}"]
    CALL -- "Exception khác" --> EXC["{ok: false, error: 'tool_exception'}<br/>bug trong tool KHÔNG được giết loop"]
    CALL -- "trả về" --> ISDICT{"isinstance(result, dict)?"}

    ISDICT -- "No" --> WRAP["{ok: true, result: result}"]
    ISDICT -- "Yes" --> PASS["trả nguyên envelope của tool<br/>(ok true/false + hints)"]

    UNKNOWN --> OUT
    BADARGS --> OUT
    EXC --> OUT
    WRAP --> OUT
    PASS --> OUT
    OUT(["luôn trả dict JSON-serializable"])

    classDef err fill:#f8d7da,stroke:#c0392b,color:#000
    classDef ok fill:#d4edda,stroke:#2e7d32,color:#000
    class UNKNOWN,BADARGS,EXC err
    class PASS,WRAP ok
```

Mỗi envelope lỗi đều kèm **hint** (`available_tools`, `received`, `available_items`,
`supported_destinations`) — đó là thứ cho phép Agent tự phục hồi thay vì lặp lại đúng lời gọi hỏng.

---

## 3. V2 — repeated-action detector (khác biệt duy nhất so với V1)

```mermaid
flowchart TD
    ACT(["parse_action &rarr; (tool_name, args)"]) --> KEY["key = (tool_name, json(args, sort_keys=True))<br/>đảo thứ tự key KHÔNG lách được"]
    KEY --> SEEN{"key đã có trong seen_calls ?"}

    SEEN -- "No — lần đầu" --> RUN["_execute_tool(...)<br/>tool_calls += 1<br/>seen_calls[key] = observation"]
    RUN --> APPEND["nối Observation vào transcript"]

    SEEN -- "Yes — trùng" --> NORUN["<b>KHÔNG chạy tool</b><br/>repeats += 1<br/>log REPEATED_ACTION"]
    NORUN --> BUDGET{"repeats &gt; repeat_budget ?<br/>(mặc định 1)"}

    BUDGET -- "No — cảnh báo lần đầu" --> WARN["observation = {error: 'repeated_action',<br/>previous_observation: &lt;kết quả cache&gt;,<br/>'reply with Final Answer now'}"]
    WARN --> APPEND

    BUDGET -- "Yes — lặp lần hai" --> STOP["termination = 'repeated_action'<br/>break<br/>REPEAT_FALLBACK + evidence đã thu"]

    classDef v2 fill:#e7d9ff,stroke:#6b3fa0,color:#000
    classDef ok fill:#d4edda,stroke:#2e7d32,color:#000
    class SEEN,NORUN,BUDGET,WARN,STOP v2
    class RUN ok
```

V1 chạy lại mọi action nó nhận được; lời gọi trùng trả observation trùng nên transcript không có thêm
thông tin — livelock chỉ `max_steps` phá được. V2 chặn ngay tại đó. Đo trên cùng script, `max_steps=8`:
`llm_calls` 8→5, `tool_calls` 8→3, lời gọi trùng bị thực thi 5→**0**, tokens 960→600.

---

## 4. Chatbot Baseline vs ReAct Agent

```mermaid
flowchart LR
    subgraph CB["Chatbot Baseline — src/chatbot/chatbot.py"]
        direction TB
        C1(["user_input"]) --> C2["system_prompt + history + input"]
        C2 --> C3["1 lần gọi LLM"]
        C3 --> C4(["answer<br/>llm_calls=1, tool_calls=0"])
    end

    subgraph AG["ReAct Agent — src/agent/agent.py"]
        direction TB
        A1(["user_input"]) --> A2["system_prompt + TOOL_SPECS + transcript"]
        A2 --> A3["gọi LLM"]
        A3 --> A4["Action &rarr; chạy tool thật"]
        A4 --> A5["Observation (bằng chứng)"]
        A5 --> A3
        A3 --> A6(["Final Answer<br/>llm_calls=N, tool_calls=M"])
    end

    CB -.->|"cùng 1 câu hỏi,<br/>cùng 1 model"| AG

    classDef bad fill:#f8d7da,stroke:#c0392b,color:#000
    classDef good fill:#d4edda,stroke:#2e7d32,color:#000
    class C4 bad
    class A6 good
```

Baseline có `tool_calls == 0` **về mặt cấu trúc**, nên với case multi-step nó chỉ còn hai kết cục
trung thực: thú nhận không biết, hoặc bịa số. Agent đi qua vòng Observation nên đến được ground truth
**45.038.000 VND** (2 × 25.000.000 − 10% + 38.000 phí ship).

---

## 5. Trace mẫu — case multi-step (đường đi lý tưởng)

```mermaid
sequenceDiagram
    autonumber
    participant U as User
    participant A as ReActAgent
    participant L as LLM
    participant T as Tools

    U->>A: "2 iPhone, mã WINNER, ship Hà Nội, 0.8kg. Tổng?"
    A->>L: transcript (step 1)
    L-->>A: Thought + Action: check_stock({"item_name":"iPhone"})
    A->>T: check_stock("iPhone")
    T-->>A: {ok:true, price:25000000, stock:15}
    Note over A: (I3) nối Observation vào transcript

    A->>L: transcript + Observation (step 2)
    L-->>A: Action: get_discount({"coupon_code":"WINNER"})
    A->>T: get_discount("WINNER")
    T-->>A: {ok:true, discount_percent:10, valid:true}

    A->>L: transcript + Observation (step 3)
    L-->>A: Action: calc_shipping({"weight":0.8,"destination":"Hanoi"})
    A->>T: calc_shipping(0.8, "Hanoi")
    T-->>A: {ok:true, shipping_cost:38000, estimated_days:1}

    A->>L: transcript + Observation (step 4)
    L-->>A: Thought + Final Answer: 45.038.000 VND
    A-->>U: answer — llm_calls=4, tool_calls=3, termination=final_answer
```
