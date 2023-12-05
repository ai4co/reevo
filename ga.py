from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
import re
import shutil
import subprocess
import numpy as np

from utils.utils import file_to_string, block_until_running, filter_traceback


class GA_LLM:
    def __init__(self, cfg, root_dir): # TODO semantic niche
        self.client = OpenAI()
        self.cfg = cfg
        self.root_dir = root_dir
        
        self.iteration = 0
        self.population = self.init_population()
        
        
    def init_population(self):
        self.problem = self.cfg.problem.problem_name
        self.problem_description = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_description)
        
        prompt_dir = f'{self.root_dir}/utils/prompts_{self.cfg.problem_type}'
        problem_dir = f"{self.root_dir}/problems/{self.problem}"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{self.cfg.suffix.lower()}.py"
        
        # Loading all text prompts
        initial_system = file_to_string(f'{prompt_dir}/initial_system.txt')
        initial_user = file_to_string(f'{prompt_dir}/initial_user.txt')
        code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        func_signature = file_to_string(f'{problem_dir}/func_signature.txt')
        initial_system = initial_system.format(func_signature=func_signature) + code_output_tip
        initial_user = initial_user.format(problem_description=self.problem_description)

        # Generate responses
        messages = [{"role": "system", "content": initial_system}, {"role": "user", "content": initial_user}]
        responses = self.chat_completion(self.cfg, self.client, messages)
        
        # Run code and evaluate population
        population, objs = self.evaluate_population(responses)
        
        # Bookkeeping
        self.best_obj_overall, best_sample_idx = min(objs), np.argmin(np.array(objs))
        self.best_code_overall = population[best_sample_idx]["code"]
        self.best_desc_overall = population[best_sample_idx]["description"]
        self.best_code_path_overall = population[best_sample_idx]["code_path"]

        logging.info(f"Iteration {self.iteration}: Min obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        self.iteration += 1
        return population
        
    def evaluate_population(self, responses) -> tuple[list[dict], list[float]]:
        code_runs = []
        descriptions_runs = []
        inner_runs = []
        for response_id in range(self.cfg.sample):
            response = responses[response_id].message.content
            code_string, desc_string, process = self.run_code(response, response_id)
            code_runs.append(code_string)
            descriptions_runs.append(desc_string)
            inner_runs.append(process)
        # Gather population
        population = []
        objs = []
        for response_id, (code_run, descriptions_run, rl_run) in enumerate(zip(code_runs, descriptions_runs, inner_runs)):
            rl_run.communicate()
            
            # Initialize individual
            individual = {
                "stdout_filepath": f"problem_iter{self.iteration}_response{response_id}.txt",
                "code_path": f"problem_iter{self.iteration}_response{response_id}.py",
                "description": descriptions_run,
                "code": code_run,
            }
            stdout_filepath = f"problem_iter{self.iteration}_response{response_id}.txt"
            with open(stdout_filepath, 'r') as f:
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            if traceback_msg == '': # If execution has no error
                individual["exec_success"] = True
                obj = float(stdout_str.split('\n')[-2])
                individual["obj"] = obj
            else:                   # Otherwise, also provide execution traceback error feedback
                individual["traceback_msg"] = traceback_msg
                individual["exec_success"] = False
                obj = float("inf")
                individual["obj"] = obj
            objs.append(obj)
            population.append(individual)
        return population, objs

        
    
    def run_code(self, response, response_id) -> tuple[str, str, subprocess.Popen]:
        """
        Extract code and its description from the responses.
        Write code into a file.
        Run eval script.
        """
        logging.info(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        # Regex patterns to extract python code enclosed in GPT response
        pattern_code = r'```python(.*?)```'
        code_string = re.search(pattern_code, response, re.DOTALL).group(1).strip()
        # Regex patterns to extract code description enclosed in GPT response
        pattern_desc = r'Code description: (.*)\n'
        desc_string = re.search(pattern_desc, response).group(1).strip()
        logging.debug(f"Iteration {self.iteration}: GPT description:\n " + desc_string)
        logging.debug(f"Iteration {self.iteration}: GPT code:\n " + code_string)
        
        with open(self.output_file, 'w') as file:
            file.writelines(code_string + '\n')

        # Copy the generated code to hydra output directory for bookkeeping
        shutil.copy(self.output_file, f"problem_iter{self.iteration}_response{response_id}.py")

        # Execute the python file with flags
        stdout_filepath = f"problem_iter{self.iteration}_response{response_id}.txt"
        stdout_filepath = f"problem_iter{self.iteration}_response{response_id}.txt"
        with open(stdout_filepath, 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{self.root_dir}/problems/{self.cfg.problem.problem_name}/eval.py', f'{self.problem_size}', self.root_dir],
                                        stdout=f, stderr=f)

        block_until_running(stdout_filepath, log_status=True, iter_num=self.iteration, response_id=response_id)
        
        return code_string, desc_string, process
        
    
    def chat_completion(self, cfg, client: OpenAI, messages: list[dict]) -> list[dict]:
        responses = []
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = cfg.sample if "gpt-3.5" in cfg.model else min(4, cfg.sample)
        while True:
            if total_samples >= cfg.sample:
                break
            for attempt in range(1000):
                try:
                    response_cur = client.chat.completions.create(model=cfg.model, messages=messages, temperature=cfg.temperature, n=chunk_size)
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
        
        logging.info(f"Iteration {self.iteration}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        return responses
    
    def update_iter(self, population, objs):
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        if best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_desc_overall = population[best_sample_idx]["description"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        logging.info(f"Iteration {self.iteration}: Min obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        self.iteration += 1
    
    @staticmethod
    def compute_similarity(code_snippets: list[str], client: OpenAI, model: str="text-embedding-ada-002") -> np.ndarray:
        """
        Embed multiple code snippets using OpenAI's embedding API and compute the cosine similarity matrix.
        """
        response = client.embeddings.create(
            input=code_snippets,
            model=model,
        )
        embeddings = [_data.embedding for _data in response.data]
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix

if __name__ == '__main__':
    # Example usage
    code_snippets = [
        "def add(a, b): return a + b",
        "def subtract(a, b): return a - b",
        "def multiply(a, b): return a * b"
    ]

