from .openai import OpenAIClient
from zhipuai import ZhipuAI 

class ZhipuAIClient(OpenAIClient):
    ClientClass = ZhipuAI

    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature,
        )