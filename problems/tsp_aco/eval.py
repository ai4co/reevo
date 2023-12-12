from aco import ACO
import sys
import numpy as np
from scipy.spatial import distance_matrix

from gpt import scoring_function


N_ITERATIONS = 100
N_ANTS = 30


if __name__ == "__main__":
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    
    dataset_path = f"{root_dir}/problems/tsp_aco/dataset/val{problem_size}_dataset.npy"
    node_positions = np.load(dataset_path)
    n_instances = node_positions.shape[0]
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
    
    objs = []
    for i, node_pos in enumerate(node_positions):
        dist_mat = distance_matrix(node_pos, node_pos)
        dist_mat[np.diag_indices_from(dist_mat)] = np.inf # Note: Set diagonal to inf
        heuristics = scoring_function(dist_mat) + 1e-9
        aco = ACO(dist_mat, heuristics, n_ants=N_ANTS)
        obj = aco.run(N_ITERATIONS)
        print(f"[*] Instance {i}: {obj}")
        objs.append(obj)
    
    print("[*] Average:")
    print(np.mean(objs))