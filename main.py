import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import re
import subprocess
from pathlib import Path
import shutil
import time 
from pprint import pprint

from utils.utils import * 

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ROOT_DIR = os.getcwd()

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")

    problem = cfg.problem.problem_name
    problem_size = cfg.problem.problem_size
    problem_description = cfg.problem.description
    suffix = cfg.suffix
    model = cfg.model
    logging.info(f"Using LLM: {model}")
    logging.info("Problem: " + problem)
    logging.info("Problem description: " + problem_description)

    output_file = f"{ROOT_DIR}/problems/{problem}/{suffix.lower()}.py"

    # Loading all text prompts
    prompt_dir = f'{ROOT_DIR}/utils/prompts_{cfg.problem_type}'
    problem_dir = f"{ROOT_DIR}/problems/{problem}"
    initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
    code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
    initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
    func_signature = file_to_string(f'{problem_dir}/func_signature.txt')
    # example = file_to_string(f'{problem_dir}/example.txt')
    optimization_feedback = file_to_string(f'{prompt_dir}/optimization_feedback.txt')
    execution_error_feedback = file_to_string(f'{prompt_dir}/execution_error_feedback.txt')

    initial_system = initial_system.format(func_signature=func_signature) + code_output_tip
    initial_user = initial_user.format(problem_description=problem_description)
    messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]

    
    best_obj_overall = float('inf')
    
    # Generation loop
    for iter in range(cfg.iteration):
        # pprint(messages)
        
        # Get response
        responses = []
        response_cur = None
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in model else 4

        logging.info(f"Iteration {iter}: Generating {cfg.sample} samples with {cfg.model}")

        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = client.chat.completions.create(model=model, messages=messages, temperature=cfg.temperature, n=chunk_size)
                    total_samples += chunk_size
                    break
                except Exception as e:
                    if attempt >= 10:
                        chunk_size = max(int(chunk_size / 2), 1)
                        print("Current Chunk Size", chunk_size)
                    logging.info(f"Attempt {attempt+1} failed with error: {e}")
                    time.sleep(1)
            if response_cur is None:
                logging.info("Code terminated due to too many failed attempts!")
                exit()

            responses.extend(response_cur.choices)
            prompt_tokens = response_cur.usage.prompt_tokens
            total_completion_token += response_cur.usage.completion_tokens
            total_token += response_cur.usage.total_tokens

        # Logging Token Information
        logging.info(f"Iteration {iter}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        
        code_runs = []
        response_runs = []
        inner_runs = []
        for response_id in range(cfg.sample):
            response_cur = responses[response_id].message.content
            logging.info(f"Iteration {iter}: GPT Output:\n " + response_cur)
            logging.info(f"Iteration {iter}: Processing Code Run {response_id}")

            # Regex patterns to extract python code enclosed in GPT response
            pattern = r'```python(.*?)```'
            code_string = re.search(pattern, response_cur, re.DOTALL)
            if code_string is not None:
                code_string = code_string.group(1).strip()
            
            code_runs.append(code_string)
            response_runs.append(response_cur)

            with open(output_file, 'w') as file:
                file.writelines(code_string + '\n')

            # Copy the generated code to hydra output directory for bookkeeping
            shutil.copy(output_file, f"problem_iter{iter}_response{response_id}.py")

            # Execute the python file with flags
            stdout_filepath = f"problem_iter{iter}_response{response_id}.txt"
            with open(stdout_filepath, 'w') as f:
                process = subprocess.Popen(['python', '-u', f'{ROOT_DIR}/problems/{problem}/eval.py', f'{problem_size}', ROOT_DIR],
                                            stdout=f, stderr=f)

            block_until_running(stdout_filepath, log_status=True, iter_num=iter, response_id=response_id)
            inner_runs.append(process)
        
        # Gather results
        contents = []
        code_paths = []
        objs = []
        exec_success = False 
        for response_id, (code_run, rl_run) in enumerate(zip(code_runs, inner_runs)):
            rl_run.communicate()
            stdout_filepath = f"problem_iter{iter}_response{response_id}.txt"
            code_paths.append(f"problem_iter{iter}_response{response_id}.py")

            with open(stdout_filepath, 'r') as f:
                stdout_str = f.read() 

            content = ''
            traceback_msg = filter_traceback(stdout_str)
            
            if traceback_msg == '':
                # If execution has no error, provide statistics feedback
                exec_success = True
                obj = float(stdout_str.split('\n')[-2])
                objs.append(obj) # the smaller the better
                content += optimization_feedback
                
            else:
                # Otherwise, provide execution traceback error feedback
                objs.append(float('inf'))
                content += execution_error_feedback.format(traceback_msg=traceback_msg)

            content += code_output_tip
            contents.append(content) 
            

        # Select the best code sample
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        best_code = code_runs[best_sample_idx]
        best_content = contents[best_sample_idx]
        
        # Update the overall best
        if best_obj < best_obj_overall:
            best_obj_overall = best_obj
            best_code_path = code_paths[best_sample_idx]


        logging.info(f"Iteration {iter}: Min obj: {best_obj}, Best Code Path: {best_code_path}")
        # logging.info(f"Iteration {iter}: GPT Output Content:\n" +  responses[best_sample_idx]["message"]["content"] + "\n")
            
        # if len(messages) == 2:
        #     messages += [{"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}]
        #     messages += [{"role": "user", "content": best_content}]
        # else:
        #     assert len(messages) == 4
        #     messages[-2] = {"role": "assistant", "content": responses[best_sample_idx]["message"]["content"]}
        #     messages[-1] = {"role": "user", "content": best_content}

if __name__ == "__main__":
    main()