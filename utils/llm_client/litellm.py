from .base import BaseClient
import logging

try:
    from litellm import completion
except ImportError:
    completion = None

logger = logging.getLogger(__name__)

class LiteLLMClient(BaseClient):
    
    def __init__(
        self, 
        model: str, 
        temperature: float = 1,
    ) -> None:
        
        super().__init__(model, temperature)

        if completion is None:
            logging.fatal(f"Package `litellm` is required")
            exit(-1)
        
        from litellm import validate_environment
        validity = validate_environment(model)
        if not validity['keys_in_environment']:
            logger.fatal(f"Missing environment variables: {repr(validity['missing_keys'])}")
            exit(-1)
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        assert n == 1
        response = completion(model=self.model, messages=messages, temperature=temperature)
        return response.choices