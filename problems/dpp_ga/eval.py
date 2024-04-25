import random
import time
from os import path

# from time import time

import numpy as np
from reward_functions import RewardModel

from tqdm import tqdm

seed = 5678
random.seed("%d" % (seed))


def seed_crossover(parents: np.ndarray, n_pop: int) -> np.ndarray:
    """Seed crossover."""
    n_parents, n_decap = parents.shape

    # Split genomes into two halves
    left_halves = parents[:, :n_decap // 2]
    right_halves = parents[:, n_decap // 2:]

    # Create parent pairs
    parents_idx = np.stack([np.random.choice(range(n_parents), 2, replace=False) for _ in range(n_pop)])
    parents_left = left_halves[parents_idx[:, 0]]
    parents_right = right_halves[parents_idx[:, 1]]

    # Create offspring
    offspring = np.concatenate([parents_left, parents_right], axis=1)
    return offspring


try:
    from gpt import crossover_v2 as crossover
except:
    from gpt import crossover



def generate_population(population_size: int, n_decap: int, probe: float, prohibit: np.ndarray, n: int, m: int) -> np.ndarray:
    # Create the full range of actions, excluding 'probe' and any 'prohibit' values
    possible_actions = np.setdiff1d(np.arange(n * m), np.append(prohibit, probe))
   # Ensure that the possible actions can fill the required number of decaps
    if len(possible_actions) < n_decap:
        raise ValueError("Not enough valid actions to fill the individuals without replacement.")
    # Randomly select 'n_decap' unique actions from the possible actions
    pop = np.stack([np.random.choice(possible_actions, n_decap, replace=False) for _ in range(population_size)])
    return pop



def mutation(population: np.ndarray, probe: float, prohibit: np.ndarray) -> None:
    """Mutation while validating the population."""
    n_pop, n_decap = population.shape
    for i in range(n_pop):
        ind = population[i]
        unique_actions = np.unique(population[i])
        if len(unique_actions) < n_decap:
            # Find the indices wherein the action is taken the second time
            dup_idx = []
            action_set = set()
            for j, action in enumerate(ind):
                if action in action_set:
                    dup_idx.append(j)
                action_set.add(action)

            # Mutate the duplicated actions
            infeasible_actions = np.concatenate([prohibit, [probe], unique_actions])
            feasible_actions = np.setdiff1d(np.arange(n * m), infeasible_actions)
            assert n_decap - len(unique_actions) == len(dup_idx)
            new_actions = np.random.choice(feasible_actions, len(dup_idx), replace=False)
            ind[dup_idx] = new_actions
    return population

def check_feasibility(population: np.ndarray, probe: float, prohibit: np.ndarray) -> None:
    """Check if the population is feasible."""
    return # skip this check for efficiency
    n_pop, n_decap = population.shape
    for i in range(n_pop):
        unique_actions = np.unique(population[i])
        if len(unique_actions) < n_decap:
            raise ValueError("Population is infeasible.")
        for action in population[i]:
            if action in prohibit or action == probe:
                raise ValueError("Population is infeasible.")

    
def eval_population(population, probe, reward_model) -> np.ndarray:
    rewards = [
        reward_model(probe, pi)
        for pi in population
        ]
    return np.array(rewards)


def run_ga(n_pop: int, n_iter: int, n_inst: int, elite_rate: float, n_decap: int, reward_model: RewardModel) -> float:
    """
    Runs the Genetic Algorithm (GA) for optimization.

    Args:
        n_pop (int): Population size.
        n_iter (int): Number of generations.
        n_inst (int): Number of test instances.
        elite_rate (float): Percentage of elite individuals.
        n_decap (int): Number of decap.
        reward_model (RewardModel): Reward model for scoring the individuals.
    """
    sum_reward = 0

    # Outer loop: test instances
    for j in tqdm(range(n_inst), desc="Testing {} instances".format(n_inst), disable=True):
        start_time = time.time()

        probe = int(test_probe[j])
        prohibit = test_prohibit[j]
        keep_num = int(keepout_num[j])
        prohibit = prohibit[0: keep_num]

        population = generate_population(n_pop, n_decap, probe, prohibit, n, m)  # shape: (P, n x m)
        rewards = eval_population(population, probe, reward_model)  # shape: (P,)
        print("Initial population avg. reward:", rewards.mean())
        # Inner loop: generations
        for i in range(n_iter):
            # Sort the population and rewards according to the reward
            sorted_idx = rewards.argsort() # ascending order
            population = population[sorted_idx]
            rewards = rewards[sorted_idx]
            
            # Select the better half of the population
            better_half = population[int(n_pop / 2):]

            # Preserve the elites
            n_elite = int(n_pop * elite_rate)
            elites = better_half[-n_elite:]

            # Crossover with the better half
            population_nxt = crossover(better_half, n_pop=n_pop - n_elite)
            # Mutate the population
            population_nxt = mutation(population_nxt, probe, prohibit)

            # Check the feasibility of the next generation
            check_feasibility(population_nxt, probe, prohibit)
            
            # Evaluate the population
            rewards_nxt = eval_population(population_nxt, probe, reward_model)

            # Elitism
            # 1. Concate the elites
            population = np.concatenate([elites, population_nxt], axis=0)
            # 2. Concate the rewards
            rewards = np.concatenate([rewards[-n_elite:], rewards_nxt], axis=0)

            print("Generation {:d} - Elite reward: {:.4f}".format(i, rewards[:n_elite].mean()) + " - Best reward: {:.4f}".format(rewards.max()))


        # Evaluate the final population
        best_idx = np.argmax(rewards)
        best_solution, best_reward = population[best_idx], rewards[best_idx]
        sum_reward += best_reward
        print("Best solution:", best_solution)
        print("Best reward:", best_reward)
        print("--- %s seconds ---" % (time.time() - start_time))

    # result = plot_result.plot(raw_pdn, probe, guide_action, n, m, j)
    print("Average reward:", sum_reward / n_inst)
    
    return sum_reward / n_inst


if __name__ == "__main__":
    import sys
    import os
    print("[*] Running ...")
        
    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]


    # Parameters
    n = 10 # PDN shape
    m = 10 # PDN shape
    model = 5 # Reward model type
    freq_pts = 201 # Number of Frequencies

    base_path = path.dirname(__file__)

    # Paths
    test_probe_path = os.path.join(base_path, "test_problems", "test_100_probe.npy")
    test_prohibit_path = os.path.join(base_path, "test_problems", "test_100_keepout.npy")
    keepout_num_path = os.path.join(base_path, "test_problems", "test_100_keepout_num.npy")

    # Model initialization
    reward_model = RewardModel(base_path, n=n, m=m, model_number=model, freq_pts=freq_pts)

    # File reading
    with open(test_probe_path, "rb") as f:
        test_probe = np.load(f)  # shape (test,)

    with open(test_prohibit_path, "rb") as f1:
        test_prohibit = np.load(f1)  # shape (test, n_keepout)

    with open(keepout_num_path, "rb") as f2:
        keepout_num = np.load(f2)  # shape (test,)


    n_pop = 20
    n_iter = 5
    elite_rate = 0.2
    n_decap = 20
    
    if mood == 'train':
        n_inst = 3
        test_probe = test_probe[0: 3]
        test_prohibit = test_prohibit[0: 3]
        keepout_num = keepout_num[0: 3]
        avg_reward = run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
        
        print("[*] Average:")
        print(avg_reward)
    
    elif mood == 'val':
        n_inst = 5
        test_probe = test_probe[5: 10]
        test_prohibit = test_prohibit[5: 10]
        keepout_num = keepout_num[5: 10]
        avg_reward = run_ga(n_pop, n_iter, n_inst, elite_rate, n_decap, reward_model)
        
        print("[*] Average:")
        print(avg_reward)


