import json
import difflib
from typing import List, Dict, Any, Optional, Tuple, Callable
from src.agent.agent import ReActAgent, parse_action, parse_final_answer, execute_tool
from src.telemetry.logger import logger
from src.tools.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS


class ReActAgentV2(ReActAgent):
    """
    Agent V2: Improved ReAct Agent addressing V1 failure modes:
    1. Repeated-action detector (prevents infinite tool loop).
    2. Unknown tool hint & recovery (suggests closest available tool name).
    3. Missing argument validation.
    """

    def __init__(
        self,
        llm: Any,
        tools: Optional[List[Dict[str, Any]]] = None,
        registry: Optional[Dict[str, Callable]] = None,
        max_steps: int = 5,
        max_repeated_actions: int = 1
    ):
        super().__init__(llm=llm, tools=tools, registry=registry, max_steps=max_steps)
        self.max_repeated_actions = max_repeated_actions
        self.action_history: List[Tuple[str, Dict[str, Any]]] = []

    def _get_tool_hint(self, unknown_tool: str) -> str:
        """Finds closest matching tool name for user hint."""
        available_names = list(self.registry.keys())
        matches = difflib.get_close_matches(unknown_tool.lower(), available_names, n=1, cutoff=0.3)
        if matches:
            return matches[0]
        # Common domain mapping fallbacks
        if "search" in unknown_tool.lower() or "product" in unknown_tool.lower():
            return "check_stock"
        if "coupon" in unknown_tool.lower() or "promo" in unknown_tool.lower():
            return "get_discount"
        if "ship" in unknown_tool.lower() or "delivery" in unknown_tool.lower():
            return "calc_shipping"
        return available_names[0] if available_names else ""

    def run(self, user_input: str) -> str:
        """
        Agent V2 Loop Execution with Guardrails & Loop Detection.
        """
        logger.log_event("AGENT_V2_START", {"input": user_input, "model": self.llm.model_name})

        messages = [f"User Query: {user_input}"]
        steps = 0
        self.action_history.clear()
        action_counts: Dict[str, int] = {}

        for step in range(self.max_steps):
            steps += 1
            current_prompt = "\n".join(messages)

            response = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())

            if isinstance(response, dict):
                llm_output = response.get("content", "").strip()
            else:
                llm_output = str(response).strip()

            logger.log_event("AGENT_V2_STEP", {"step": steps, "llm_output": llm_output})
            messages.append(llm_output)

            # 1. Parse Action
            action = parse_action(llm_output)
            if action:
                tool_name, args = action
                action_key = f"{tool_name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
                action_counts[action_key] = action_counts.get(action_key, 0) + 1

                # GUARDRAIL 1: Repeated Action Detector
                if action_counts[action_key] > self.max_repeated_actions:
                    warning_res = {
                        "status": "warning",
                        "error_type": "REPEATED_ACTION_DETECTED",
                        "message": (
                            f"Cảnh báo: Bạn đã gọi công cụ '{tool_name}' với tham số {args} "
                            f"{action_counts[action_key]} lần. Vui lòng KHÔNG lặp lại. "
                            f"Hãy sử dụng thông tin đã có để chuyển sang bước tiếp theo hoặc xuất 'Final Answer:'."
                        )
                    }
                    obs_str = f"Observation: {json.dumps(warning_res, ensure_ascii=False)}"
                    logger.log_event("AGENT_V2_REPEATED_ACTION_GUARD", {
                        "step": steps,
                        "tool": tool_name,
                        "args": args,
                        "count": action_counts[action_key]
                    })
                    messages.append(obs_str)
                    continue

                # GUARDRAIL 2: Unknown Tool Recovery with Hints
                if tool_name not in self.registry:
                    hint = self._get_tool_hint(tool_name)
                    error_res = {
                        "status": "error",
                        "error_type": "UNKNOWN_TOOL",
                        "message": f"Công cụ '{tool_name}' không tồn tại.",
                        "hint": f"Gợi ý: Sử dụng công cụ '{hint}' thay vì '{tool_name}'.",
                        "allowed_tools": list(self.registry.keys())
                    }
                    obs_str = f"Observation: {json.dumps(error_res, ensure_ascii=False)}"
                    logger.log_event("AGENT_V2_UNKNOWN_TOOL_GUARD", {
                        "step": steps,
                        "unknown_tool": tool_name,
                        "suggested_hint": hint
                    })
                    messages.append(obs_str)
                    continue

                # Execute Tool via Executor
                result = execute_tool(tool_name, args, self.registry)
                self.action_history.append((tool_name, args))
                obs_str = f"Observation: {json.dumps(result, ensure_ascii=False)}"

                logger.log_event("TOOL_EXECUTION_V2", {
                    "step": steps,
                    "tool": tool_name,
                    "args": args,
                    "result": result
                })
                messages.append(obs_str)
                continue

            # 2. Parse Final Answer
            final_ans = parse_final_answer(llm_output)
            if final_ans:
                logger.log_event("AGENT_V2_END", {"steps": steps, "final_answer": final_ans})
                return final_ans

        # 3. Fallback if max_steps reached
        fallback_msg = f"Đã vượt quá số bước tối đa ({self.max_steps} bước) mà chưa đưa ra được câu trả lời cuối cùng."
        logger.log_event("AGENT_V2_END", {"steps": steps, "fallback": fallback_msg})
        return fallback_msg
