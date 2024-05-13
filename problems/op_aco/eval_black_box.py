from aco import ACO
import numpy as np
import logging
from gen_inst import OPInstance, load_dataset
import torch
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 50
N_ANTS = 20


def solve(inst: OPInstance):
    heu = heuristics(np.array(inst.prize), np.array(inst.distance), inst.maxlen) + 1e-9
    assert tuple(heu.shape) == (inst.n, inst.n)
    heu[heu < 1e-9] = 1e-9
    heu = torch.from_numpy(heu)
    aco = ACO(inst.prize, inst.distance, inst.maxlen, heu, N_ANTS)
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
        dataset = load_dataset(dataset_path)
        n_instances = len(dataset)

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj.item())
        
        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in [50, 100, 200]:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
            dataset = load_dataset(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(dataset):
                obj = solve(instance)
                objs.append(obj.item())
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")