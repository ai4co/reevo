from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
import time
import re
import subprocess
import numpy as np

from utils.utils import file_to_string, block_until_running, filter_traceback


class GA_LLM:
    def __init__(self, cfg, root_dir) -> None:
        self.client = OpenAI()
        self.cfg = cfg
        self.root_dir = root_dir
        
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float("inf")
        
        self.invalid_responses = 0 # Number of invalid responses
        self.total_responses = 0 # Number of total responses
        
        self.init_prompt()
        
        if cfg.diversify:
            while True:
                self.greedy_obj = self.evaluate_greedy_alg()
                if self.greedy_obj != float("inf"):
                    break

        self.init_population()

        if cfg.problem_type != "constructive":
            assert cfg.diversify == False, "Diversification is not supported for problem types other than constructive."

            logging.info(f"Greedy Algorithm Objective Value: {self.greedy_obj}")
        
        self.ga_crossover_prompt = file_to_string(f'{root_dir}/utils/prompts_ga/crossover.txt')
        self.ga_mutate_prompt = file_to_string(f'{root_dir}/utils/prompts_ga/mutate.txt')
        
        self.print_cross_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration

    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_description = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_description)
        
        prompt_dir = f'{self.root_dir}/utils/prompts_{self.cfg.problem_type}'
        problem_dir = f"{self.root_dir}/problems/{self.problem}"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{self.cfg.suffix.lower()}.py"
        
        # Loading all text prompts
        func_signature = file_to_string(f'{problem_dir}/func_signature.txt')
        self.code_output_tip = file_to_string(f'{prompt_dir}/code_output_tip.txt')
        self.system_prompt = file_to_string(f'{prompt_dir}/system.txt').format(func_signature=func_signature)
        self.initial_user = file_to_string(f'{prompt_dir}/initial_user.txt').format(problem_description=self.problem_description)

        
    def init_population(self) -> None:
        # Generate responses
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.initial_user + self.code_output_tip}]
        logging.info("Initial prompt: \nSystem Prompt: \n" + self.system_prompt + "\nUser Prompt: \n" + self.initial_user + self.code_output_tip)
        responses = self.chat_completion(self.cfg.pop_size, self.cfg, messages)
        
        # Responses to population
        population = self.responses_to_population(responses)
        
        # Run code and evaluate population
        population = self.evaluate_population(population)
        objs = [individual["obj"] for individual in population]
        
        # Bookkeeping
        self.best_obj_overall, best_sample_idx = min(objs), np.argmin(np.array(objs))
        self.best_code_overall = population[best_sample_idx]["code"]
        self.best_desc_overall = population[best_sample_idx]["description"]
        self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # Update iteration
        self.population = population
        self.update_iter()

    
    def evaluate_greedy_alg(self) -> float:
        """
        Generate and evaluate the greedy algorithm for the problem, e.g. Nearest Neighbor for TSP.
        """
        # Loading all text prompts
        greedy_alg_tip = file_to_string(f'{self.root_dir}/utils/prompts_ga/gen_greedy_tip.txt')
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.initial_user + greedy_alg_tip}]
        logging.info("Greedy Algorithm Prompt: \nSystem Prompt: \n" + self.system_prompt + "\nUser Prompt: \n" + self.initial_user + greedy_alg_tip)
        
        # Generate responses
        responses = self.chat_completion(1, self.cfg, messages)
        
        # Response to individual
        individual = self.response_to_individual(responses[0], 0, file_name="greedy_alg")
        
        # Run code and evaluate population
        population = self.evaluate_population([individual])
        return population[0]["obj"]

    
    def response_to_individual(self, response, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        self.total_responses += 1
        content = response.message.content
        
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(content + '\n')

        # Extract code and description from response
        code_string, desc_string = self.extract_code_description(content)
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"

        if code_string is None:
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Extract None; invalid response!")
            self.invalid_responses += 1
        elif desc_string is None:
            desc_string = "This is an algorithm that solves the problem." # Default description
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Extract no description; use default description.")
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "description": desc_string,
            "code": code_string,
            "response": content,
            "response_id": response_id,
        }

        return individual

        
    def responses_to_population(self, responses) -> list[dict]:
        """
        Convert responses to population. Applied to the initial population.
        """
        population = []
        for response_id, response in enumerate(responses):
            individual = self.response_to_individual(response, response_id)
            population.append(individual)
        return population

    @staticmethod
    def mark_invalid_individual(individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["fitness"] = 0
        individual["traceback_msg"] = traceback_msg
        return individual

    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values and fitness.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            self.function_evals += 1
            
            try:
                process = self.run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values and fitness
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=20) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            
            individual = population[response_id]
            # Store objective value and fitness for each individual
            if traceback_msg == '': # If execution has no error
                individual["exec_success"] = True
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2])
                    assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                    individual["fitness"] = 1 / individual["obj"]
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    def extract_code_description(self, response: str) -> tuple[str, str]:
        # Regex patterns to extract python code enclosed in GPT response
        pattern_code = r'```python(.*?)```'
        code_string = re.search(pattern_code, response, re.DOTALL)
        code_string = code_string.group(1).strip() if code_string is not None else None
        # Regex patterns to extract code description enclosed in GPT response
        pattern_desc = r'Code description: (.*?)```python'
        desc_string = re.search(pattern_desc, response, re.DOTALL)
        desc_string = desc_string.group(1).strip() if desc_string is not None else None
        return code_string, desc_string


    def run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{self.root_dir}/problems/{self.problem}/eval.py', f'{self.problem_size}', self.root_dir],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process


    def chat_completion(self, n: int, cfg, messages: list[dict]) -> list[dict]:
        """
        Generate n responses using OpenAI Chat Completions API
        """
        responses = []
        total_samples = 0
        total_token = 0
        total_completion_token = 0
        chunk_size = n if "gpt-3.5" in cfg.model else min(4, n)
        while True:
            if total_samples >= n:
                break
            for attempt in range(1000):
                try:
                    response_cur = self.client.chat.completions.create(model=cfg.model, messages=messages, temperature=cfg.temperature, n=chunk_size)
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
        
        logging.debug(f"Iteration {self.iteration}: Prompt Tokens: {prompt_tokens}, Completion Tokens: {total_completion_token}, Total Tokens: {total_token}")
        return responses

    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        
        # update best overall
        if best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_desc_overall = population[best_sample_idx]["description"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            # If diversification is enabled, only update elitist if the best individual is better than the greedy algorithm
            if not self.cfg.diversify or population[best_sample_idx]["obj"] < self.greedy_obj:
                self.elitist = population[best_sample_idx]
                logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Min obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        logging.info(f"Invalid Responses: {self.invalid_responses}, Total Responses: {self.total_responses}")
        self.iteration += 1


    @staticmethod
    def fitness_sharing(similarity_matrix: np.ndarray, fitness: list[float], sigma_share: float=0.9) -> list[float]:
        """
        Fitness sharing based on cosine similarity matrix to encourage diversity in the population.
        
        :param similarity_matrix: An n x n matrix representing the cosine similarity between each pair of individuals.
        :param fitness: A list of original fitness values for each individual.
        :param sigma_share: The sharing threshold, defining the niche size.
        :param alpha: A constant that determines the shape of the sharing function.
        :return: A list of adjusted fitness values.
        """

        n = len(fitness)
        adjusted_fitness = np.zeros(n)

        # Define the sharing function adjusted for cosine similarity
        def sharing_function(similarity: float) -> float:
            if similarity > sigma_share: 
                return (similarity - sigma_share) / (1 - sigma_share)
            else:
                return 0
        
        # Calculate the niche count for each individual
        for i in range(n):
            m_i = sum(sharing_function(similarity_matrix[i][j]) for j in range(n))
            adjusted_fitness[i] = fitness[i] / m_i if m_i != 0 else fitness[i]

        return adjusted_fitness.tolist()
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability. Deprecated.
        """
        selected_population = []
        for _ in range(self.cfg.pop_size):
            parents = np.random.choice(population, size=2, replace=False)
            selected_population.extend(parents)
        assert len(selected_population) == 2*self.cfg.pop_size
        return selected_population


    def select(self, population: list[dict]) -> list[dict]:
        """
        Roulette selection, select individuals with probability proportional to their adjusted fitness
        """
        # Eliminate those without description (description is None or "")
        population = [individual for individual in population if individual["description"] is not None and individual["description"] != ""]
        
        similarity_matrix = self.compute_similarity([individual["description"] for individual in population], self.client)
        similarity_matrix = self.adjust_similarity_matrix(similarity_matrix, population)
        # logging.info("Similarity Matrix: \n" + str(similarity_matrix))
        
        fitness = [individual["fitness"] for individual in population]
        # logging.info("Fitness before sharing: \n" + str(fitness))
        
        fitness = self.fitness_sharing(similarity_matrix, fitness)
        logging.info("Fitness after sharing: \n" + str(fitness))
        
        fitness_sum = sum(fitness)
        fitness_prob = [f / fitness_sum for f in fitness]
        selected_population = []
        for _ in range(self.cfg.pop_size):
            parents = np.random.choice(population, size=2, p=fitness_prob, replace=False) # 2x population size for crossover
            selected_population.extend(parents)
        
        assert len(selected_population) == 2*self.cfg.pop_size
        return selected_population


    def crossover(self, population: list[dict]) -> list[dict]:
        crossed_population = []
        assert len(population) == self.cfg.pop_size * 2
        response_id = 0
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Crossover
            crossover_prompt_user = self.ga_crossover_prompt.format(
                problem_description=self.problem_description,
                code1=parent_1["code"],
                code2=parent_2["code"],
                description1=parent_1["description"],
                description2=parent_2["description"],
                )
            messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": crossover_prompt_user + self.code_output_tip}]
            
            # Print crossover prompt for the first iteration
            if self.print_cross_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + self.system_prompt + "\nUser Prompt: \n" + crossover_prompt_user + self.code_output_tip)
                self.print_cross_prompt = False
            
            # Generate response and convert response to individual
            responses = self.chat_completion(1, self.cfg, messages)
            individual = self.response_to_individual(responses[0], response_id)
            crossed_population.append(individual)
            response_id += 1

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def mutate(self, population: list[dict]) -> list[dict]:
        for i in range(len(population)):
            individual = population[i]
            
            # Mutate
            if np.random.uniform() < self.cfg.mutation_rate:
                mutate_prompt = self.ga_mutate_prompt.format(
                    problem_description=self.problem_description,
                    code=individual["code"],
                    description=individual["description"]
                    )
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": mutate_prompt + self.code_output_tip}]
                
                # Print mutate prompt for the first iteration
                if self.print_mutate_prompt:
                    logging.info("Mutate Prompt: \nSystem Prompt: \n" + self.system_prompt + "\nUser Prompt: \n" + mutate_prompt + self.code_output_tip)
                    self.print_mutate_prompt = False
                
                # Generate response and convert response to individual
                responses = self.chat_completion(1, self.cfg, messages)
                mutated_individual = self.response_to_individual(responses[0], individual["response_id"])
                population[i] = mutated_individual
        assert len(population) == self.cfg.pop_size
        return population


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # Diversify
            if self.cfg.diversify: self.diversify()
            # Select
            population_to_select = self.population if self.elitist is None else [self.elitist] + self.population # add elitist to population for selection
            selected_population = self.select(population_to_select)
            # Crossover
            crossed_population = self.crossover(selected_population)
            # Mutate
            mutated_population = self.mutate(crossed_population)
            # Evaluate
            population = self.evaluate_population(mutated_population)
            # Update
            self.population = population
            self.update_iter()
        return self.best_code_overall, self.best_desc_overall, self.best_code_path_overall


    @staticmethod
    def compute_similarity(descriptions: list[str], client: OpenAI, model: str="text-embedding-ada-002") -> np.ndarray:
        """
        Embed code descriptions using OpenAI's embedding API and compute the cosine similarity matrix.
        """
        response = client.embeddings.create(
            input=descriptions,
            model=model,
        )
        embeddings = [_data.embedding for _data in response.data]
        similarity_matrix = cosine_similarity(embeddings)
        return similarity_matrix


    @staticmethod
    def adjust_similarity_matrix(similarity_matrix: np.ndarray, population: list[dict]) -> np.ndarray:
        """
        For those with identical objective values, set their similarity to 1.
        """
        for i in range(len(population)):
            for j in range(i+1, len(population)):
                if population[i]["obj"] == population[j]["obj"]:
                    similarity_matrix[i][j] = 1
                    similarity_matrix[j][i] = 1
        return similarity_matrix



    def diversify(self):
        """
        Diversify the population by eliminating the greedy algorithms (obj == self.greedy_obj) and adding new ones.
        """
        # Eliminate greedy algorithms or those with execution errors
        preserved_population = [individual for individual in self.population if individual["obj"] != self.greedy_obj and individual["exec_success"]]
        n_eliminated = self.cfg.pop_size - len(preserved_population)
        logging.info(f"Eliminated {n_eliminated} greedy or invalid algorithms.")
        
        # Return if no new individuals are needed
        if n_eliminated == 0:
            self.update_iter()
            return
        
        # Generate new responses
        messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": self.initial_user + self.code_output_tip}]
        responses = self.chat_completion(n_eliminated, self.cfg, messages)
        
        # Responses to population
        new_population = self.responses_to_population(responses)
        
        # Run code and evaluate population
        new_population = self.evaluate_population(new_population)
        
        # Add new population to population
        self.population = preserved_population + new_population
        self.update_iter()
