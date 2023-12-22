from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import logging
import subprocess
import numpy as np

from utils.utils import *


class G2A:
    def __init__(self, cfg, root_dir) -> None:
        self.client = OpenAI()
        self.cfg = cfg
        self.root_dir = root_dir
        
        self.iteration = 0
        self.function_evals = 0
        self.elitist = None
        self.best_obj_overall = float("inf")
        
        self.init_prompt()
        self.init_population()


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_description = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        
        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_description)
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/{self.cfg.suffix.lower()}.py"
        
        # Loading all text prompts
        self.generator_system_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.reflector_system_prompt = file_to_string(f'{self.prompt_dir}/common/system_reflector.txt')
        self.task_desc = file_to_string(f'{self.prompt_dir}/{self.problem}/task_desc.txt')
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/{self.problem}/seed.txt')
        self.user_reflector_prompt = file_to_string(f'{self.prompt_dir}/{self.problem}/user_reflector.txt')
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/{self.problem}/crossover.txt')
        
        self.print_cross_prompt = True # Print crossover prompt for the first iteration
        self.print_mutate_prompt = True # Print mutate prompt for the first iteration
        self.print_short_term_reflection_prompt = True # Print short-term reflection prompt for the first iteration


    def init_population(self) -> None:
        # Generate responses
        system = self.generator_system_prompt
        user = self.task_desc + self.seed_prompt
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
        responses = chat_completion(self.cfg.pop_size, messages, self.cfg.model, self.cfg.temperature)
        
        # Responses to population
        population = self.responses_to_population(responses)
        
        # Run code and evaluate population
        population = self.evaluate_population(population)
        objs = [individual["obj"] for individual in population]
        
        # Bookkeeping
        self.best_obj_overall, best_sample_idx = min(objs), np.argmin(np.array(objs))
        self.best_code_overall = population[best_sample_idx]["code"]
        self.best_code_path_overall = population[best_sample_idx]["code_path"]

        # Update iteration
        self.population = population
        self.update_iter()

    
    def response_to_individual(self, response, response_id, file_name=None) -> dict:
        """
        Convert response to individual
        """
        content = response.message.content
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(content + '\n')

        code = extract_code_from_generator(content)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"
        
        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
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
                try:
                    individual["obj"] = float(stdout_str.split('\n')[-2])
                    assert individual["obj"] > 0, "Objective value <= 0 is not supported."
                    individual["fitness"] = 1 / individual["obj"]
                    individual["exec_success"] = True
                except:
                    population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid std out / objective value!")
            else: # Otherwise, also provide execution traceback error feedback
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)

            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


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
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Min obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
    
    
    def random_select(self, population: list[dict]) -> list[dict]:
        """
        Random selection, select individuals with equal probability.
        """
        selected_population = []
        # Eliminate invalid individuals
        population = [individual for individual in population if individual["exec_success"]]
        while len(selected_population) < 2 * self.cfg.pop_size:
            parents = np.random.choice(population, size=2, replace=False)
            # If two parents have the same objective value, consider them as identical; otherwise, add them to the selected population
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
        return selected_population

    def gen_short_term_reflection_prompt(self, ind1: dict, ind2: dict):
        """
        Short-term reflection before crossovering two individuals.
        """
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1
        else:
            raise ValueError("Two individuals to crossover have the same objective value!")
        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])
        message = [{"role": "system", "content": self.reflector_system_prompt},
                   {"role": "user", "content": self.user_reflector_prompt.format(worse_code=worse_code, better_code=better_code)}]
        return message, worse_code, better_code
    
    
    def short_term_reflection(self, population: list[dict]):
        """
        Short-term reflection before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term reflection
            messages, worse_code, better_code = self.gen_short_term_reflection_prompt(parent_1, parent_2)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
            
            # Print reflection prompt for the first iteration
            if self.print_short_term_reflection_prompt:
                logging.info("Short-term Reflection Prompt: \nSystem Prompt: \n" + self.reflector_system_prompt + "\nUser Prompt: \n" + self.user_reflector_prompt.format(worse_code=parent_2["code"], better_code=parent_1["code"]))
                self.print_short_term_reflection_prompt = False
        
        # Multi-processed chat completion
        responses_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        return responses_lst, worse_code_lst, better_code_lst


    def crossover(self, population: list[dict]) -> list[dict]:
        crossed_population = []
        assert len(population) == self.cfg.pop_size * 2
        
        # Short-term reflection
        responses_lst, worse_code_lst, better_code_lst = self.short_term_reflection(population)
        
        messages_lst = []
        
        idx = 0
        for response, worse_code, better_code in zip(responses_lst, worse_code_lst, better_code_lst):
            reflection = response[0].message.content
            
            # Crossover
            crossover_prompt_user = self.crossover_prompt.format(
                task_desc = self.task_desc,
                worse_code = worse_code,
                better_code = better_code,
                reflection = reflection,
            )
            messages = [{"role": "system", "content": self.generator_system_prompt}, {"role": "user", "content": crossover_prompt_user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_cross_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + self.generator_system_prompt + "\nUser Prompt: \n" + crossover_prompt_user)
                self.print_cross_prompt = False
            
            # Save crossover prompt to file
            file_name = f"problem_iter{self.iteration}_crossover{idx}.txt"
            with open(file_name, 'w') as file:
                file.writelines(crossover_prompt_user + '\n')
            idx += 1
        
        # Multi-processed chat completion
        responses_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
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
            if np.random.uniform() < self.cfg.mutation_rate:
                mutate_prompt = self.ga_mutate_prompt.format(
                    problem_description=self.problem_description,
                    code=individual["code"],
                    description=individual["description"]
                    )
                messages = [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": mutate_prompt}]
                messages_lst.append(messages)
                response_id_lst.append(i)
                # Print mutate prompt for the first iteration
                if self.print_mutate_prompt:
                    logging.info("Mutate Prompt: \nSystem Prompt: \n" + self.system_prompt + "\nUser Prompt: \n" + mutate_prompt)
                    self.print_mutate_prompt = False
            
        # Multi-processed chat completion
        responses_lst = multi_chat_completion(messages_lst, 1, self.cfg.model, self.cfg.temperature)
        for i in range(len(responses_lst)):
            response_id = response_id_lst[i]
            mutated_individual = self.response_to_individual(responses_lst[i][0], response_id)
            population[response_id] = mutated_individual

        assert len(population) == self.cfg.pop_size
        return population


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            # Select
            population_to_select = self.population if self.elitist is None else [self.elitist] + self.population # add elitist to population for selection
            selected_population = self.random_select(population_to_select)
            # Crossover
            crossed_population = self.crossover(selected_population)
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            # Update
            self.update_iter()
            # Mutate
            # mutated_population = self.mutate(crossed_population)
            # # Evaluate
            # self.population = self.evaluate_population(mutated_population)
            # # Update
            # self.update_iter()

        return self.best_code_overall, self.best_code_path_overall
