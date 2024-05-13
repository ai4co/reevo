import os
from aco import ACO
import sys
import numpy as np
from scipy.spatial import distance_matrix
import logging
import inspect
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)

N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50

def solve(node_pos, demand):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    if len(inspect.getfullargspec(heuristics).args) == 4:
        heu = heuristics(dist_mat.copy(), node_pos.copy(), demand.copy(), CAPACITY) + 1e-9
    elif len(inspect.getfullargspec(heuristics).args) == 2:
        heu = heuristics(dist_mat.copy(), demand / CAPACITY) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, demand, heu, CAPACITY, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']

    basepath = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(basepath, "dataset/train50_dataset.npy")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
        dataset = np.load(dataset_path)
        demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
        
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
            obj = solve(node_pos, demand)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj.item())
        
        print("[*] Average:")
        print(np.mean(objs))
        
    else:
        for problem_size in [20, 50, 100]:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npy")
            dataset = np.load(dataset_path)
            demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
            
            n_instances = node_positions.shape[0]
            logging.info(f"[*] Evaluating {dataset_path}")
            
            objs = []
            for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
                obj = solve(node_pos, demand)
                objs.append(obj.item())
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")