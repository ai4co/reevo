from aco import ACO
import sys
import numpy as np
from scipy.spatial import distance_matrix
import logging

from gpt import heuristics_v2 as heuristics


N_ITERATIONS = 100
N_ANTS = 30


def solve(node_pos):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1 # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
    return obj

if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    
    if mood == 'train':
        dataset_path = f"{root_dir}/problems/tsp_aco/dataset/{mood}{problem_size}_dataset.npy"
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, node_pos in enumerate(node_positions):
            obj = solve(node_pos)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))
    
    else:
        for problem_size in [20, 50, 100]:
            dataset_path = f"{root_dir}/problems/tsp_aco/dataset/{mood}{problem_size}_dataset.npy"
            node_positions = np.load(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")
            n_instances = node_positions.shape[0]
            objs = []
            for i, node_pos in enumerate(node_positions):
                obj = solve(node_pos)
                objs.append(obj.item())
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")