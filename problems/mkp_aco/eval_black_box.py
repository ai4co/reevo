from aco import ACO
import numpy as np
import torch
import logging
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 50
N_ANTS = 10

def solve(prize: np.ndarray, weight: np.ndarray):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS)
    obj, _ = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train50_dataset.npz")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
        dataset = np.load(dataset_path)
        prizes, weights = dataset['prizes'], dataset['weights']
        n_instances = prizes.shape[0]

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, (prize, weight) in enumerate(zip(prizes, weights)):
            obj = solve(prize, weight)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj.item())
        
        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in [100, 300, 500]:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
            dataset = np.load(dataset_path)
            prizes, weights = dataset['prizes'], dataset['weights']
            n_instances = prizes.shape[0]
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, (prize, weight) in enumerate(zip(prizes, weights)):
                obj = solve(prize, weight)
                objs.append(obj.item())
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")