from .openai import OpenAIClient

try:
    from zhipuai import ZhipuAI
except ImportError:
    ZhipuAI = "zhipuai"

class ZhipuAIClient(OpenAIClient):

    ClientClass = ZhipuAI

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=min(temperature, 1.0),
        )
        return response.choices