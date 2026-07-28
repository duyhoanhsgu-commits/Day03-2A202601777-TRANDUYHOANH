import os
from typing import Dict, Any, Optional
from src.core.llm_provider import LLMProvider
from src.telemetry.logger import logger

DEFAULT_ECOMMERCE_SYSTEM_PROMPT = """Bạn là một Trợ lý Chăm sóc Khách hàng Thương mại điện tử thông minh tại TechMart, một cửa hàng trực tuyến chuyên về điện tử, điện thoại thông minh, máy tính xách tay và phụ kiện.

### TRÁCH NHIỆM CỦA BẠN:
1. **Hỗ trợ & Tư vấn Sản phẩm**: Cung cấp câu trả lời thân thiện, chính xác và chuyên nghiệp cho các câu hỏi của khách hàng về thông số kỹ thuật sản phẩm, khuyến nghị và giá cả.

2. **Đơn hàng & Vận chuyển**: Hướng dẫn khách hàng cách kiểm tra trạng thái đơn hàng, phương thức vận chuyển (Tiêu chuẩn: 2-3 ngày, Nhanh: 1 ngày) và phí giao hàng.

3. **Khuyến mãi & Giảm giá**: Giúp khách hàng hiểu các mã giảm giá đang hoạt động (ví dụ: 'WELCOME10' giảm 10% cho người dùng mới, 'TECH50' giảm $50 cho đơn hàng trên $500).

4. **Chính sách Trả hàng & Hoàn tiền**: Thông báo cho khách hàng về chính sách trả hàng trong 30 ngày đối với các mặt hàng bị lỗi và phạm vi bảo hành.

### QUY TẮC ỨNG XỬ:
- **Giọng điệu**: Luôn giữ thái độ lịch sự, nhiệt tình, hữu ích và ngắn gọn.
- **Ngôn ngữ**: Trả lời bằng cùng ngôn ngữ với người dùng (tiếng Việt hoặc tiếng Anh).
- **Hạn chế**: Bạn KHÔNG có quyền truy cập vào các công cụ cơ sở dữ liệu trực tuyến hoặc tra cứu đơn hàng cá nhân theo thời gian thực. Nếu người dùng yêu cầu thông tin cá nhân theo thời gian thực (ví dụ: "Đơn hàng số 12345 của tôi hiện đang ở đâu?"), hãy lịch sự thông báo cho họ về các hướng dẫn chung và đề nghị liên hệ với bộ phận hỗ trợ hoặc sử dụng trang theo dõi đơn hàng tự phục vụ.
- **Không bịa đặt**: Không được bịa đặt số lượng hàng tồn kho hoặc số theo dõi hàng giả theo thời gian thực.
"""

class EcommerceChatbot:

    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: Optional[str] = None
    ):
        self.llm = llm
        self.system_prompt = system_prompt or DEFAULT_ECOMMERCE_SYSTEM_PROMPT

    def run(self, user_input: str) -> Dict[str, Any]:
        """
        Processes user input and returns the chatbot's response along with telemetry data.
        """
        logger.log_event("CHATBOT_START", {
            "input": user_input,
            "model": self.llm.model_name
        })

        # Generate completion from LLM provider
        result = self.llm.generate(
            prompt=user_input,
            system_prompt=self.system_prompt
        )

        logger.log_event("CHATBOT_END", {
            "model": self.llm.model_name,
            "usage": result.get("usage"),
            "latency_ms": result.get("latency_ms")
        })

        return result

    def chat(self, user_input: str) -> str:
        """
        Simple wrapper that returns only the message content string.
        """
        res = self.run(user_input)
        return res.get("content", "")


if __name__ == "__main__":
    from dotenv import load_dotenv
    from src.core.gemini_provider import GeminiProvider

    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY không tồn tại.")
    else:
        provider = GeminiProvider(model_name="gemini-1.5-flash", api_key=api_key)
        chatbot = EcommerceChatbot(llm=provider)
        
        test_query = "Tôi muốn mua iPhone 15 và dùng mã giảm giá TECH50 thì được giảm bao nhiêu?"
        print(f"User: {test_query}\n")
        response = chatbot.chat(test_query)
        print(f"Chatbot: {response}")
