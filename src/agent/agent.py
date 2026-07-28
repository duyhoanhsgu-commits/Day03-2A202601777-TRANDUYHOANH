import os
import re
import ast
import json
from typing import List, Dict, Any, Optional, Tuple, Callable
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger
from src.tools.tools import AVAILABLE_TOOLS, TOOL_SCHEMAS


# Mapping parameter names for known tools to support positional argument conversion
TOOL_PARAM_NAMES: Dict[str, List[str]] = {
    "check_stock": ["item_name"],
    "get_discount": ["coupon_code"],
    "calc_shipping": ["weight", "destination"],
}


def _convert_val(val_str: str) -> Any:
    """Helper to convert string representations of numbers/booleans to native Python types."""
    val_clean = val_str.strip().strip("'\"")
    if val_clean.lower() == "true":
        return True
    if val_clean.lower() == "false":
        return False
    try:
        if "." in val_clean:
            return float(val_clean)
        return int(val_clean)
    except ValueError:
        return val_clean


def parse_action(text: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Parses Action string from LLM output.
    Supports formats:
    - Action: tool_name(key1="val1", key2=1.5)
    - Action: tool_name("arg1", "arg2")
    - Action: {"name": "tool_name", "args": {...}}
    - Action: tool_name({"key": "val"})

    Returns tuple[str, dict] or None if Action is not found.
    """
    if not text:
        return None

    # Format 1: Action: {"name": "tool_name", "args": {...}}
    json_action_match = re.search(r"Action:\s*(\{.*\})", text, re.DOTALL)
    if json_action_match:
        try:
            data = json.loads(json_action_match.group(1).strip())
            if isinstance(data, dict) and "name" in data:
                tool_name = data["name"]
                args = data.get("args", {})
                if not isinstance(args, dict):
                    args = {"arg": args}
                return tool_name, args
        except Exception:
            pass

    # Format 2: Action: tool_name(arguments)
    action_match = re.search(r"Action:\s*([a-zA-Z0-9_]+)\s*\((.*?)\)", text, re.DOTALL)
    if not action_match:
        return None

    tool_name = action_match.group(1).strip()
    raw_args = action_match.group(2).strip()

    if not raw_args:
        return tool_name, {}

    # Attempt AST parsing e.g. dummy(weight=1.5, destination="Hà Nội")
    try:
        call_ast = ast.parse(f"dummy({raw_args})", mode="eval")
        if isinstance(call_ast.body, ast.Call):
            positional_args = []
            kwargs = {}
            for arg in call_ast.body.args:
                positional_args.append(ast.literal_eval(arg))
            for kw in call_ast.body.keywords:
                kwargs[kw.arg] = ast.literal_eval(kw.value)

            if positional_args:
                param_names = TOOL_PARAM_NAMES.get(tool_name, [])
                for i, val in enumerate(positional_args):
                    if i < len(param_names):
                        kwargs[param_names[i]] = val
                    else:
                        kwargs[f"arg_{i}"] = val
            return tool_name, kwargs
    except Exception:
        pass

    # Fallback parsing if ast literal_eval fails
    kwargs = {}
    positional_args = []
    parts = [p.strip() for p in raw_args.split(",") if p.strip()]

    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            kwargs[k.strip()] = _convert_val(v)
        else:
            positional_args.append(_convert_val(part))

    if positional_args:
        param_names = TOOL_PARAM_NAMES.get(tool_name, [])
        for i, val in enumerate(positional_args):
            if i < len(param_names):
                kwargs[param_names[i]] = val
            else:
                kwargs[f"arg_{i}"] = val

    return tool_name, kwargs


def parse_final_answer(text: str) -> Optional[str]:
    """
    Parses 'Final Answer: <response>' from LLM text output.
    Returns content string or None if Final Answer is not found.
    """
    if not text or "Final Answer:" not in text:
        return None
    return text.split("Final Answer:", 1)[1].strip()


def execute_tool(tool_name: str, args: Dict[str, Any], registry: Dict[str, Callable]) -> Dict[str, Any]:
    """
    Executes a tool from the registry safely.
    Validates tool existence, passes arguments, wraps any exception in a structured JSON dict.
    """
    if tool_name not in registry:
        return {
            "status": "error",
            "error_type": "TOOL_NOT_FOUND",
            "message": f"Công cụ '{tool_name}' không tồn tại trong hệ thống. Các công cụ khả dụng: {list(registry.keys())}"
        }

    func = registry[tool_name]
    try:
        if isinstance(args, dict):
            result = func(**args)
        elif isinstance(args, (list, tuple)):
            result = func(*args)
        else:
            result = func(args)

        if isinstance(result, dict):
            return result
        return {"status": "success", "result": result}

    except TypeError as te:
        return {
            "status": "error",
            "error_type": "INVALID_ARGUMENTS",
            "message": f"Lỗi tham số khi gọi tool '{tool_name}': {str(te)}"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": "EXECUTION_ERROR",
            "message": f"Lỗi khi thực thi tool '{tool_name}': {str(e)}"
        }


class ReActAgent:
    """
    ReAct Agent following Thought-Action-Observation loop with Parser and Executor modules.
    """

    def __init__(
        self,
        llm: LLMProvider,
        tools: Optional[List[Dict[str, Any]]] = None,
        registry: Optional[Dict[str, Callable]] = None,
        max_steps: int = 5
    ):
        self.llm = llm
        self.tools = tools if tools is not None else TOOL_SCHEMAS
        self.registry = registry if registry is not None else AVAILABLE_TOOLS
        self.max_steps = max_steps
        self.history = []

    def get_system_prompt(self) -> str:
        """
        Phần A — System Prompt
        Bao gồm: Vai trò, Danh sách tool + Mô tả + Ví dụ input, Output format, Quy tắc không invent tool, Xử lý error, Điều kiện dừng.
        """
        tool_desc_list = []
        for t in self.tools:
            tool_desc_list.append(f"- {t['name']}: {t['description']}")
        tool_descriptions = "\n".join(tool_desc_list)

        return f"""Bạn là một Trợ lý AI Chăm sóc Khách hàng thông minh tại cửa hàng điện tử TechMart.
Nhiệm vụ của bạn là suy luận từng bước và sử dụng công cụ phù hợp để giải quyết câu hỏi của người dùng.

### DANH SÁCH CÔNG CỤ (TOOLS KHẢ DỤNG):
{tool_descriptions}

### VÍ DỤ CÁCH GỌI TOOL (INPUT EXAMPLES):
- Action: check_stock(item_name="iPhone")
- Action: get_discount(coupon_code="WINNER")
- Action: calc_shipping(weight=1.5, destination="Hà Nội")

### ĐỊNH DẠNG ĐẦU RA (OUTPUT FORMAT):
Hãy tuân thủ nghiêm ngặt quy trình ReAct sau:
Thought: Suy luận từng bước của bạn về thông tin cần tìm hoặc bước xử lý tiếp theo.
Action: tool_name(key1="val1", key2="val2")
Observation: Kết quả trả về từ công cụ (Hệ thống sẽ cung cấp sau khi Action chạy).
... (Lặp lại Thought/Action/Observation nếu cần thiết)
Thought: Tôi đã có đủ dữ liệu để đưa ra kết luận cuối cùng.
Final Answer: Câu trả lời chi tiết, chính xác và đầy đủ gửi tới người dùng.

### NGUYÊN TẮC VÀ ĐIỀU KIỆN DỪNG:
1. KHÔNG INVENT TOOL: Chỉ gọi các công cụ có trong danh sách trên. Không tự bịa ra công cụ mới.
2. XỬ LÝ LỖI: Nếu Observation trả về lỗi ("status": "error"), hãy xem xét lại tham số hoặc thông báo rõ cho người dùng.
3. KHÔNG TỰ BỊA OBSERVATION: Đợi hệ thống thực thi công cụ và trả về kết quả Observation thực tế.
4. ĐIỀU KIỆN DỪNG: Dừng ngay lập tức khi xuất ra 'Final Answer: ...' hoặc đạt tối đa số bước cho phép ({self.max_steps} bước).
"""

    def run(self, user_input: str) -> str:
        """
        Phần B — Loop Pattern
        Executes Thought-Action-Observation loop up to max_steps.
        """
        logger.log_event("AGENT_START", {"input": user_input, "model": self.llm.model_name})

        messages = [f"User Query: {user_input}"]
        steps = 0

        for step in range(self.max_steps):
            steps += 1
            current_prompt = "\n".join(messages)
            
            response = self.llm.generate(current_prompt, system_prompt=self.get_system_prompt())
            
            if isinstance(response, dict):
                llm_output = response.get("content", "").strip()
            else:
                llm_output = str(response).strip()

            logger.log_event("AGENT_STEP", {"step": steps, "llm_output": llm_output})
            messages.append(llm_output)

            # 1. Parse Action
            action = parse_action(llm_output)
            if action:
                tool_name, args = action
                # Executor: run tool, wrap exception into structured JSON
                result = execute_tool(tool_name, args, self.registry)
                obs_str = f"Observation: {json.dumps(result, ensure_ascii=False)}"
                
                logger.log_event("TOOL_EXECUTION", {
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
                logger.log_event("AGENT_END", {"steps": steps, "final_answer": final_ans})
                return final_ans

        # 3. Safe fallback if max_steps exceeded
        fallback_msg = f"Đã vượt quá số bước tối đa ({self.max_steps} bước) mà chưa đưa ra được câu trả lời cuối cùng."
        logger.log_event("AGENT_END", {"steps": steps, "fallback": fallback_msg})
        return fallback_msg

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """
        Legacy helper method for backward compatibility.
        """
        parsed_action = parse_action(f"Action: {tool_name}({args})")
        if parsed_action:
            _, parsed_args = parsed_action
        else:
            parsed_args = {}
        res = execute_tool(tool_name, parsed_args, self.registry)
        return json.dumps(res, ensure_ascii=False)
