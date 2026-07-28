import os
import sys
import pytest
import json

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.agent.agent import ReActAgent, parse_action, parse_final_answer, execute_tool
from src.core.llm_provider import LLMProvider
from src.tools.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS, check_stock, get_discount, calc_shipping


class ScriptedLLM(LLMProvider):
    """
    Scripted LLM Provider to simulate multi-step ReAct agent reasoning
    without making external API calls.
    """
    def __init__(self, responses: list[str]):
        super().__init__(model_name="scripted-react-llm", api_key="mock")
        self.responses = responses
        self._iter = iter(responses)

    def generate(self, prompt: str, system_prompt: str = None) -> dict:
        try:
            content = next(self._iter)
        except StopIteration:
            content = "Final Answer: (Scripted responses exhausted)"
            
        return {
            "content": content,
            "usage": {"prompt_tokens": 120, "completion_tokens": 60, "total_tokens": 180},
            "latency_ms": 80,
            "provider": "scripted"
        }

    def stream(self, prompt: str, system_prompt: str = None):
        yield self.generate(prompt, system_prompt)["content"]


def test_react_loop_e2e_multistep():
    """
    Test Case: Complete E2E ReAct Agent loop using ScriptedLLM.
    Query: 2 iPhones + WINNER discount + Hanoi shipping.
    Steps:
      1. check_stock("iPhone")
      2. get_discount("WINNER")
      3. calc_shipping(weight=1.0, destination="Hà Nội")
      4. Final Answer
    """
    scripted_responses = [
        "Thought: Tôi cần kiểm tra giá và hàng tồn kho của sản phẩm iPhone.\nAction: check_stock(item_name=\"iPhone\")",
        "Thought: iPhone có giá 20,000,000 VND. Tiếp theo tôi sẽ kiểm tra mã giảm giá WINNER.\nAction: get_discount(coupon_code=\"WINNER\")",
        "Thought: Mã WINNER hợp lệ và giảm 10%. Tiếp theo tôi tính phí vận chuyển đến Hà Nội.\nAction: calc_shipping(weight=1.0, destination=\"Hà Nội\")",
        "Thought: Đã thu thập đầy đủ dữ liệu. Giá 2 chiếc iPhone là 40,000,000 VND, giảm 10% (4,000,000 VND) còn 36,000,000 VND. Phí vận chuyển Hà Nội là 30,000 VND. Tổng cộng 36,030,000 VND.\nFinal Answer: Tổng chi phí mua 2 chiếc iPhone áp dụng mã WINNER và giao tới Hà Nội là 36,030,000 VND."
    ]

    llm = ScriptedLLM(scripted_responses)
    agent = ReActAgent(llm=llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=5)

    user_query = "Tôi muốn mua 2 chiếc iPhone kèm mã giảm giá WINNER và giao hàng tới Hà Nội. Tổng chi phí là bao nhiêu?"
    result = agent.run(user_query)

    assert "36,030,000 VND" in result
    print("\n✅ Test E2E ReAct Agent Loop Multi-step PASSED!")


def test_react_loop_max_steps_exceeded():
    """
    Test Case: ReAct Agent reaches max_steps without returning Final Answer.
    Expected: Agent returns a safe fallback message.
    """
    infinite_loop_responses = [
        "Thought: Step 1 thinking...\nAction: check_stock(\"iPhone\")",
        "Thought: Step 2 thinking...\nAction: check_stock(\"iPhone\")",
        "Thought: Step 3 thinking...\nAction: check_stock(\"iPhone\")"
    ]

    llm = ScriptedLLM(infinite_loop_responses)
    agent = ReActAgent(llm=llm, tools=TOOL_SCHEMAS, registry=AVAILABLE_TOOLS, max_steps=2)

    result = agent.run("Infinite loop test query")
    assert "Đã vượt quá số bước tối đa" in result
    print("✅ Test Max Steps Fallback PASSED!")


def test_parser_action_and_final_answer():
    """
    Test Case: Verify parse_action and parse_final_answer utility functions.
    """
    # 1. Action with kwargs
    action1 = parse_action('Action: calc_shipping(weight=2.5, destination="TP.HCM")')
    assert action1 == ("calc_shipping", {"weight": 2.5, "destination": "TP.HCM"})

    # 2. Action with positional string
    action2 = parse_action('Action: check_stock("MacBook")')
    assert action2 == ("check_stock", {"item_name": "MacBook"})

    # 3. Action with JSON payload
    action3 = parse_action('Action: {"name": "get_discount", "args": {"coupon_code": "TECH50"}}')
    assert action3 == ("get_discount", {"coupon_code": "TECH50"})

    # 4. Final Answer
    final_ans = parse_final_answer("Thought: Done.\nFinal Answer: Phí vận chuyển là 20,000 VND.")
    assert final_ans == "Phí vận chuyển là 20,000 VND."

    print("✅ Test Parser Functions PASSED!")


def test_executor_error_handling():
    """
    Test Case: Verify execute_tool behavior with valid, invalid, and throwing tools.
    """
    # Valid tool call
    res1 = execute_tool("get_discount", {"coupon_code": "WINNER"}, AVAILABLE_TOOLS)
    assert res1["discount_percent"] == 10
    assert res1["valid"] is True

    # Unknown tool call
    res2 = execute_tool("unknown_tool", {}, AVAILABLE_TOOLS)
    assert res2["status"] == "error"
    assert res2["error_type"] == "TOOL_NOT_FOUND"

    # Tool throwing exception
    def faulty_tool():
        raise RuntimeError("Service unavailable!")

    res3 = execute_tool("faulty", {}, {"faulty": faulty_tool})
    assert res3["status"] == "error"
    assert res3["error_type"] == "EXECUTION_ERROR"
    assert "Service unavailable!" in res3["message"]

    print("✅ Test Executor Error Handling PASSED!")


if __name__ == "__main__":
    pytest.main(["-v", __file__])
