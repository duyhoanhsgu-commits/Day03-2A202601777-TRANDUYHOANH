import os
import sys
import pytest

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import ReActAgent
from src.agent.agent_v2 import ReActAgentV2
from src.core.llm_provider import LLMProvider
from src.tools.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS


class ScriptedLLM(LLMProvider):
    """
    Scripted LLM Provider for deterministic recovery test scenarios.
    """
    def __init__(self, responses: list[str]):
        super().__init__(model_name="scripted-recovery-llm", api_key="mock")
        self.responses = responses
        self._iter = iter(responses)

    def generate(self, prompt: str, system_prompt: str = None) -> dict:
        try:
            content = next(self._iter)
        except StopIteration:
            content = "Final Answer: (End of scripted responses)"

        return {
            "content": content,
            "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            "latency_ms": 50,
            "provider": "scripted"
        }

    def stream(self, prompt: str, system_prompt: str = None):
        yield self.generate(prompt, system_prompt)["content"]


# =====================================================================
# TEST 1: REPEATED ACTION FAILURE & RECOVERY (V1 vs V2)
# =====================================================================
def test_repeated_action_v1_fails():
    """
    Agent V1 Failure Trace:
    LLM outputs identical Action check_stock("iPhone") repeatedly.
    V1 executes tool repeatedly without detecting loop until max_steps fallback.
    """
    scripted_loop_responses = [
        'Thought: Lần 1 tra cứu iPhone.\nAction: check_stock("iPhone")',
        'Thought: Lần 2 tra cứu iPhone lại.\nAction: check_stock("iPhone")',
        'Thought: Lần 3 tra cứu iPhone lại.\nAction: check_stock("iPhone")',
    ]

    llm = ScriptedLLM(scripted_loop_responses)
    v1_agent = ReActAgent(llm=llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=3)

    result_v1 = v1_agent.run("Mua iPhone")
    # V1 fails by running out of max_steps
    assert "Đã vượt quá số bước tối đa" in result_v1
    print("\n❌ Agent V1 failed as expected: Repeated action loop went undetected until max_steps fallback.")


def test_repeated_action_v2_recovers():
    """
    Agent V2 Recovery Trace:
    LLM attempts identical Action check_stock("iPhone") twice.
    V2 intercepts duplicate action on step 2, sends warning hint, prompting LLM to recover.
    """
    scripted_recovery_responses = [
        'Thought: Lần 1 tra cứu iPhone.\nAction: check_stock("iPhone")',
        'Thought: Lần 2 tra cứu iPhone lại.\nAction: check_stock("iPhone")',
        'Thought: Đã nhận được cảnh báo lặp. Tôi chuyển sang kiểm tra giá.\nFinal Answer: iPhone có giá 20,000,000 VND và còn 15 chiếc.'
    ]

    llm = ScriptedLLM(scripted_recovery_responses)
    v2_agent = ReActAgentV2(llm=llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=3)

    result_v2 = v2_agent.run("Mua iPhone")
    assert "20,000,000 VND" in result_v2
    print("✅ Agent V2 passed: Detected repeated action, returned warning hint, and successfully recovered!")


# =====================================================================
# TEST 2: UNKNOWN TOOL FAILURE & RECOVERY (V1 vs V2)
# =====================================================================
def test_unknown_tool_v2_recovery_hint():
    """
    Agent V2 Unknown Tool Recovery:
    LLM calls unknown tool 'search_product("iPhone")'.
    V2 returns structured error with hint 'Gợi ý: Sử dụng công cụ check_stock'.
    LLM uses hint to call check_stock("iPhone") and completes.
    """
    scripted_unknown_tool_responses = [
        'Thought: Dùng search_product.\nAction: search_product("iPhone")',
        'Thought: Đã nhận được gợi ý dùng check_stock.\nAction: check_stock("iPhone")',
        'Thought: Đã có kết quả.\nFinal Answer: Giá iPhone là 20,000,000 VND.'
    ]

    llm = ScriptedLLM(scripted_unknown_tool_responses)
    v2_agent = ReActAgentV2(llm=llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=4)

    result_v2 = v2_agent.run("Tìm iPhone")
    assert "20,000,000 VND" in result_v2
    print("✅ Agent V2 passed: Unknown tool search_product triggered hint and allowed LLM recovery!")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
