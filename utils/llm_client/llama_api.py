import logging
from typing import Optional
from .openai import OpenAIClient
import os

logger = logging.getLogger(__name__)

class LlamaAPIClient(OpenAIClient):

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if api_key is None:
            api_key = os.getenv("LLAMA_API_KEY", None)
            assert api_key, "Please provide llama API key via environment variable LLAMA_API_KEY"
        base_url = base_url or "https://api.llama-api.com"

        super().__init__(model, temperature, base_url, api_key)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, stream=False,
            max_tokens=1024, timeout=100,
        )
        return response.choices
