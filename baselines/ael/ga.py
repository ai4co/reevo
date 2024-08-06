import logging
import subprocess
import numpy as np

from utils.utils import *


class AEL:
    def __init__(self, cfg, root_dir, client) -> None:
        self.client = client
        self.cfg = cfg
        self.root_dir = root_dir
        
        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float("inf") if cfg.problem.obj_type == "min" else -float("inf")
        
        self.invalid_responses = 0 # Number of invalid responses
        self.total_responses = 0 # Number of total responses
        
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.init_prompt()
        self.init_population()
        
        self.print_cross_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        
    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_description = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.obj_type = self.cfg.problem.obj_type
        self.problem_type = self.cfg.problem.problem_type
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_description)
        
        prompt_path_suffix = "_black_box" if self.problem_type == "black_box" else ""
        prompt_dir = f'{self.root_dir}/baselines/ael/prompts/{self.problem}{prompt_path_suffix}'
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{self.cfg.get('suffix', 'gpt').lower()}.py"
        
        # Loading all text prompts
        self.initial_prompt = file_to_string(f'{prompt_dir}/init.txt')
        self.crossover_prompt = file_to_string(f'{prompt_dir}/crossover.txt')
        self.mutate_prompt = file_to_string(f'{prompt_dir}/mutate.txt')
        
    def init_population(self) -> None:
        # Generate responses
        messages = [{"role": "user", "content": self.initial_prompt}]
        logging.info("Initial prompt: \nUser Prompt: \n" + self.initial_prompt)
        responses = self.client.chat_completion(self.cfg.pop_size, messages)
        
        # Responses to population
        population = self.responses_to_population(responses)
        
        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()

    
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
        code_string, desc_string = extract_code_from_generator(content), extract_description(content)
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"

        if code_string is None:
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Extract no code; invalid response!")
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


    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf") if self.obj_type == "min" else -float("inf")
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
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id)
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
                inner_run.communicate(timeout=10) # Wait for code execution to finish
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
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2])
                    assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                    individual["fitness"] = 1 / individual["obj"] if self.obj_type == "min" else individual["obj"]
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            process = subprocess.Popen(['python', '-u', f'{self.root_dir}/problems/{self.problem}/eval.py', f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process

    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        objs = [individual["obj"] for individual in population]
        if self.obj_type == "min":
            best_obj, best_sample_idx = min(objs), np.argmin(np.array(objs))
        else:
            best_obj, best_sample_idx = max(objs), np.argmax(np.array(objs))
        
        # update best overall
        if self.best_obj_overall is None or (self.obj_type == "min" and best_obj < self.best_obj_overall) or (self.obj_type == "max" and best_obj > self.best_obj_overall):
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_desc_overall = population[best_sample_idx]["description"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or (self.obj_type == "min" and best_obj < self.elitist["obj"]) or (self.obj_type == "max" and best_obj > self.elitist["obj"]):
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        logging.info(f"Invalid Responses: {self.invalid_responses}, Total Responses: {self.total_responses}")
        self.iteration += 1
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        population = [individual for individual in population if individual["exec_success"]]
        for _ in range(self.cfg.pop_size):
            parents = np.random.choice(population, size=2, replace=False)
            selected_population.extend(parents)
        assert len(selected_population) == 2*self.cfg.pop_size
        return selected_population

    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank selection, select individuals with probability proportional to the inverse of their rank.
        """
        selected_population = []
        # Eliminate invalid individuals
        population = [individual for individual in population if individual["exec_success"]]
        population.sort(key=lambda x: x["fitness"], reverse=True)
        # Compute rank probabilities
        rank_probs = [1 / (i+1) for i in range(len(population))]
        rank_probs = np.array(rank_probs) / np.sum(rank_probs)
        for _ in range(self.cfg.pop_size):
            parents = np.random.choice(population, size=2, replace=False, p=rank_probs)
            selected_population.extend(parents)
        assert len(selected_population) == 2*self.cfg.pop_size
        return selected_population

    def crossover(self, population: list[dict]) -> list[dict]:
        crossed_population = []
        assert len(population) == self.cfg.pop_size * 2
        
        messages_lst = []
        for i in range(0, len(population), 2):
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Crossover
            crossover_prompt_user = self.crossover_prompt.format(
                alg_desc1 = parent_1["description"],
                alg_desc2 = parent_2["description"],
                alg_code1 = parent_1["code"],
                alg_code2 = parent_2["code"],
            )
            messages = [{"role": "user", "content": crossover_prompt_user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_cross_prompt:
                logging.info("Crossover Prompt: \nUser Prompt: \n" + crossover_prompt_user)
                self.print_cross_prompt = False
        
        # Multi-processed chat completion
        responses_lst = self.client.multi_chat_completion(messages_lst)
        response_id = 0
        for i in range(len(responses_lst)):
            individual = self.response_to_individual(responses_lst[i][0], response_id)
            crossed_population.append(individual)
            response_id += 1

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def mutate(self, population: list[dict]) -> list[dict]:
        messages_lst = []
        response_id_lst = []
        for i in range(len(population)):
            individual = population[i]
            
            # Mutate
            if np.random.uniform() < self.mutation_rate:
                if individual["code"] is None:
                    continue
                mutate_prompt = self.mutate_prompt.format(
                    alg_desc = individual["description"],
                    alg_code = individual["code"],
                    )
                messages = [{"role": "user", "content": mutate_prompt}]
                messages_lst.append(messages)
                response_id_lst.append(i)
                # Print mutate prompt for the first iteration
                if self.print_mutate_prompt:
                    logging.info("Mutate Prompt: \nUser Prompt: \n" + mutate_prompt)
                    self.print_mutate_prompt = False
            
        # Multi-processed chat completion
        responses_lst = self.client.multi_chat_completion(messages_lst)
        for i in range(len(responses_lst)):
            response_id = response_id_lst[i]
            mutated_individual = self.response_to_individual(responses_lst[i][0], response_id)
            population[response_id] = mutated_individual

        assert len(population) == self.cfg.pop_size
        return population


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # Select
            selected_population = self.rank_select(self.population)
            # Crossover
            crossed_population = self.crossover(selected_population)
            # Mutate
            mutated_population = self.mutate(crossed_population)
            # Evaluate
            population = self.evaluate_population(mutated_population)
            # Update
            self.population = population
            self.update_iter()
        return self.best_code_overall, self.best_code_path_overall
