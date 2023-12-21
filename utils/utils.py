import subprocess
import os
import json
import logging
from openai import OpenAI
import multiprocessing
import time
import re


def file_to_string(filename):
    with open(filename, 'r') as file:
        return file.read()

def filter_traceback(s):
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                if "Set the environment variable HYDRA_FULL_ERROR=1" in lines[j]:
                    break
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found

def block_until_running(stdout_filepath, log_status=False, iter_num=-1, response_id=-1):
    # Ensure that the evaluation has started before moving on
    while True:
        log = file_to_string(stdout_filepath)
        if "[*] Running ..." in log or "Traceback" in log:
            if log_status and "Traceback" in log:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} execution error!")
            else:
                logging.info(f"Iteration {iter_num}: Code Run {response_id} successful!")
            break


def extract_code_description(response: str) -> tuple[str, str]:
    """Deprecated."""
    # Regex patterns to extract python code enclosed in GPT response
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, response, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    # Regex patterns to extract code description enclosed in GPT response
    pattern_desc = r'(.*?)```python'
    desc_string = re.search(pattern_desc, response, re.DOTALL)
    desc_string = desc_string.group(1).strip() if desc_string is not None else None
    return code_string, desc_string


def get_chat_completion(client, message, model="gpt-3.5-turbo-1106", temperature=0.):
    """
    Deprecated. Use chat_completion instead.
    """
    raise NotImplementedError
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=message,
        temperature=temperature,
    )
    return completion.choices[0].message.content


def multi_chat_completion(messages_list: list[list[dict]], n=1, model: str="gpt-3.5-turbo-1106", temperature: float=0.):
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
    with multiprocessing.Pool() as executor:
        # Create a list of arguments to pass to get_chat_completion
        args = [(n, messages, model, temperature) for messages in messages_list]
        # Use executor.starmap to pass the list of arguments
        contents = executor.starmap(chat_completion, args)
    return list(contents)


def chat_completion(n: int, messages: list[dict], model: str, temperature: float) -> list[dict]:
    """
    Generate n responses using OpenAI Chat Completions API
    """
    client = OpenAI()
    total_samples = 0
    responses = []
    chunk_size = n if "gpt-3.5" in model else min(4, n)
    while True:
        if total_samples >= n:
            break
        for attempt in range(1000):
            try:
                response_cur = client.chat.completions.create(model=model, messages=messages, temperature=temperature, n=min(chunk_size, n-total_samples))
                total_samples += chunk_size
                break
            except Exception as e:
                chunk_size = max(int(chunk_size / 2), 1)
                logging.info(f"Current Chunk Size: {chunk_size}")
                logging.info(f"Attempt {attempt+1} failed with error: {e}")
                time.sleep(1)
        if response_cur is None:
            logging.info("Code terminated due to too many failed attempts!")
            exit()
            
        responses.extend(response_cur.choices)
    return responses

def process_code(code):
    if code.startswith('def') and "np" in code:        
        code = 'import numpy as np\n' + code
    return code
        


if __name__ == "__main__":
    # Test multi_chat_completion
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
    responses = multi_chat_completion(messages_list, n=1, model="gpt-3.5-turbo-1106", temperature=0.)
    print(responses)