import logging
from typing import Optional
from .base import BaseClient

try:
    from openai import AzureOpenAI
except ImportError:
    AzureOpenAI = 'openai'


logger = logging.getLogger(__name__)

class AzureOpenAIClient(BaseClient):

    ClientClass = AzureOpenAI

    def __init__(
        self,
        model: Optional[str] = None,
        temperature: float = 1.0,
        endpoint: Optional[str] = None,
        deployment: Optional[str] = None,
        api_key: Optional[str] = None,
        api_version: str = "2024-12-01-preview",
        **kwargs,
    ) -> None:
        super().__init__(model, temperature)
        
        if isinstance(self.ClientClass, str):
            logger.fatal(f"Package `{self.ClientClass}` is required")
            exit(-1)
        
        self.client = self.ClientClass(
            azure_endpoint=endpoint,
            azure_deployment=deployment,
            api_key=api_key,
            api_version=api_version,
            **kwargs,
        )
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        response = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, n=n, stream=False,
        )
        return response.choices