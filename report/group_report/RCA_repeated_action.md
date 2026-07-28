# RCA — Repeated Action Livelock (Agent V1 → V2)

Paste into section 4 of `GROUP_REPORT_[TEAM].md`. Every number below is reproducible
with the commands at the bottom.

## Evidence

- Failed trace: `artifacts/traces/failure_repeated_action_v1.json`
- Run: live, `meta/llama-3.1-8b-instruct` via NVIDIA NIM, `max_steps=6`, 2026-07-28
- Regression test: `tests/test_agent_recovery.py`

## Worksheet

| Mục | Nội dung |
|-----|----------|
| **User input** | `I want to buy 2 iPhones using code 'WINNER' and ship to Hanoi. The package weight is 0.8 kg. What is the total?` |
| **Expected path** | `check_stock` → `get_discount` → `calc_shipping` → `Final Answer` |
| **Actual path** | `check_stock` → `get_discount` → `calc_shipping` → `calc_shipping` → `calc_shipping` → `calc_shipping` → safe fallback |
| **First divergence** | **Step 4.** All three facts were already in the transcript (price 25,000,000 / discount 10% / shipping 38,000). A `Final Answer` was expected; an identical `calc_shipping({"weight": 0.8, "destination": "Hanoi"})` was emitted instead. |
| **Error class** | **Loop.** Not Parser (the action parsed cleanly), not Tool (it returned `ok: true` every time), not Data. |
| **Root cause** | V1 executes whatever action it is handed. A repeated call returns a byte-identical observation, so the transcript gains no new information and the model's next state is effectively unchanged — a livelock only `max_steps` can break. Step 6 is the proof the prompt alone cannot fix this: the model wrote *"I already have the shipping fee, I should not repeat the same action"* and then repeated it. Instruction-following is not a control mechanism. |
| **Smallest fix** | Remember every `(tool, sorted args)` pair. On a repeat: do **not** run the tool, return a `repeated_action` observation carrying the cached result plus an explicit "reply with Final Answer now". On a second repeat, stop the loop. Implemented in `src/agent/agent_v2.py` — nothing else changed (parser, tool contracts, prompt wording and `max_steps` are untouched). |
| **Regression test** | `test_regression_fails_on_v1_passes_on_v2` — the same assertion is run against both agents: it raises `AssertionError` on V1 and passes on V2. |

## Before / after

Same scripted model, same question, `max_steps=8`, replaying the live failure:

| Metric | V1 | V2 | Δ |
|---|---:|---:|---|
| termination | `max_steps` | `repeated_action` | stops on cause, not on budget |
| steps | 8 | 5 | −38% |
| llm_calls | 8 | 5 | −38% |
| tool_calls executed | 8 | 3 | −63% |
| duplicate calls executed | 5 | 0 | eliminated |
| total_tokens | 960 | 600 | −38% |

Trade-off worth stating: V2 still does not answer the question. It stops earlier, spends
less and reports the evidence it holds instead of a generic "out of steps" message — but
the final arithmetic is left undone. Turning gathered evidence into a computed total is a
separate change, tied to a different divergence, and is deliberately out of scope here.

## Reproduce

```bash
python -m pytest tests/test_agent_recovery.py -q
python scripts/run_agent.py --model meta/llama-3.1-8b-instruct --save artifacts/traces/run.json
```
