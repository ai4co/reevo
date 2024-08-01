from openai import OpenAI
import os
from typing import Optional
import time
import logging
import concurrent

class OpenAIClient():

    ClientClass = OpenAI

    def __init__(
        self,
        model: str,
        temperature: float = 1.0,
        baseurl: Optional[str] = None,
        api_key: Optional[str] = None,
        api_timeout: float = 10.0,
    ) -> None:
        self.client = self.ClientClass(api_key=api_key, base_url=baseurl, timeout=api_timeout)
        self.model = model
        self.temperature = temperature
    
    def _chat_completion_api(self, messages: list[dict], temperature: float, n: int = 1):
        return self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature, n=n,
        )
    
    def chat_completion(self, n: int, messages: list[dict], temperature: Optional[float] = None) -> list[dict]:
        """
        Generate n responses using OpenAI Chat Completions API
        """
        temperature = temperature or self.temperature
        for attempt in range(1000):
            try:
                response_cur = self._chat_completion_api(messages, temperature, n)
                break
            except Exception as e:
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()
        
        return response_cur.choices
    
    def multi_chat_completion(self, messages_list: list[list[dict]], n: int = 1, temperature: Optional[float] = None):
        """
        An example of messages_list:
        
        messages_list = [
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
            ],
            [
                {"role": "system", "content": "You are a knowledgeable guide."},
                {"role": "user", "content": "How are you?"},
            ],
            [
                {"role": "system", "content": "You are a witty comedian."},
                {"role": "user", "content": "Tell me a joke."},
            ]
        ]
        param: n: number of responses to generate for each message in messages_list
        """
        # If messages_list is not a list of list (i.e., only one conversation), convert it to a list of list
        assert isinstance(messages_list, list), "messages_list should be a list."
        if not isinstance(messages_list[0], list):
            messages_list = [messages_list]
        
        if len(messages_list) > 1:
            assert n == 1, "Currently, only n=1 is supported for multi-chat completion."

        with concurrent.futures.ThreadPoolExecutor() as executor:
            args = [dict(n=n, messages=messages, temperature=temperature) for messages in messages_list]
            choices = executor.map(lambda p: self.chat_completion(**p), args)

        contents: list[str] = []
        for choice in choices:
            for c in choice:
                contents.append(c.message.content)
        return contents