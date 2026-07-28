import os
import sys
import pytest
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import parse_action, parse_final_answer, execute_tool, ReActAgent
from src.core.llm_provider import LLMProvider
from src.tools.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS


# =====================================================================
# PHẦN C — ScriptedLLM (không cần API thật)
# =====================================================================
class ScriptedLLM(LLMProvider):
    """
    Scripted LLM Provider that returns pre-defined sequential responses.
    """
    def __init__(self, responses: list[str]):
        super().__init__(model_name="scripted-llm", api_key="mock")
        self.responses = responses
        self._iter = iter(responses)

    def generate(self, prompt: str, system_prompt: str = None) -> dict:
        try:
            content = next(self._iter)
        except StopIteration:
            content = "Final Answer: (Scripted responses exhausted)"
            
        return {
            "content": content,
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            "latency_ms": 50,
            "provider": "scripted"
        }

    def stream(self, prompt: str, system_prompt: str = None):
        yield self.generate(prompt, system_prompt)["content"]


# =====================================================================
# PHẦN A TESTS — Parser Functions
# =====================================================================
def test_parse_action_positional():
    text = 'Thought: Need stock.\nAction: check_stock("iPhone")'
    action = parse_action(text)
    assert action is not None
    tool_name, args = action
    assert tool_name == "check_stock"
    assert args == {"item_name": "iPhone"}


def test_parse_action_kwargs():
    text = 'Action: calc_shipping(weight=1.5, destination="Hà Nội")'
    action = parse_action(text)
    assert action is not None
    tool_name, args = action
    assert tool_name == "calc_shipping"
    assert args["weight"] == 1.5
    assert args["destination"] == "Hà Nội"


def test_parse_action_json_format():
    text = 'Action: {"name": "get_discount", "args": {"coupon_code": "WINNER"}}'
    action = parse_action(text)
    assert action is not None
    tool_name, args = action
    assert tool_name == "get_discount"
    assert args == {"coupon_code": "WINNER"}


def test_parse_action_none():
    text = "Thought: Direct response.\nFinal Answer: Hello World!"
    assert parse_action(text) is None


def test_parse_final_answer_valid():
    text = "Thought: Done.\nFinal Answer: Total price is 36,030,000 VND."
    ans = parse_final_answer(text)
    assert ans == "Total price is 36,030,000 VND."


def test_parse_final_answer_none():
    text = "Thought: Still checking...\nAction: check_stock(\"iPhone\")"
    assert parse_final_answer(text) is None


# =====================================================================
# PHẦN B TESTS — Executor & Exception Handling
# =====================================================================
def test_execute_tool_success():
    res = execute_tool("get_discount", {"coupon_code": "WINNER"}, AVAILABLE_TOOLS)
    assert res["coupon_code"] == "WINNER"
    assert res["discount_percent"] == 10
    assert res["valid"] is True


def test_execute_tool_not_found():
    res = execute_tool("non_existent_tool", {}, AVAILABLE_TOOLS)
    assert res["status"] == "error"
    assert res["error_type"] == "TOOL_NOT_FOUND"


def test_execute_tool_exception():
    def broken_tool(x):
        raise ValueError("Database connection failed!")

    registry = {"broken_tool": broken_tool}
    res = execute_tool("broken_tool", {"x": 1}, registry)
    assert res["status"] == "error"
    assert res["error_type"] == "EXECUTION_ERROR"
    assert "Database connection failed!" in res["message"]


# =====================================================================
# PHẦN C TESTS — Loop & ScriptedLLM Integration & Trace
# =====================================================================
def test_react_loop_multistep_scripted():
    responses = [
        "Thought: Cần kiểm tra tồn kho và giá iPhone.\nAction: check_stock(\"iPhone\")",
        "Thought: Cần kiểm tra mã WINNER.\nAction: get_discount(\"WINNER\")",
        "Thought: Cần tính phí vận chuyển đến Hà Nội.\nAction: calc_shipping(weight=1.0, destination=\"Hà Nội\")",
        "Thought: Đã có đủ thông tin.\nFinal Answer: Tổng giá 2 iPhone sau khi giảm 10% và phí ship Hà Nội là 36,030,000 VND."
    ]

    scripted_llm = ScriptedLLM(responses)
    agent = ReActAgent(llm=scripted_llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS)

    final_result = agent.run("Mua 2 iPhone mã WINNER giao Hà Nội")
    assert "36,030,000 VND" in final_result


def test_react_loop_max_steps_fallback():
    # Infinite action loop without Final Answer
    responses = [
        "Thought: Loop 1.\nAction: check_stock(\"iPhone\")",
        "Thought: Loop 2.\nAction: check_stock(\"iPhone\")",
        "Thought: Loop 3.\nAction: check_stock(\"iPhone\")",
    ]

    scripted_llm = ScriptedLLM(responses)
    agent = ReActAgent(llm=scripted_llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=2)

    result = agent.run("Loop test")
    assert "Đã vượt quá số bước tối đa" in result


if __name__ == "__main__":
    pytest.main(["-v", __file__])
