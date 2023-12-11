from aco import ACO
import sys
import numpy as np
import logging
from scipy.spatial import distance_matrix

from gpt import scoring_function


N_ITERATIONS = 100
N_ANTS = 30
CAPACITY = 50


if __name__ == "__main__":
    root_dir = sys.argv[1]
    
    for problem_size in [20, 50, 100]:
    
        dataset_path = f"{root_dir}/problems/cvrp_aco/dataset/test{problem_size}_dataset.npy"
        dataset = np.load(dataset_path)
        demands, node_positions = dataset[:, :, 0], dataset[:, :, 1:]
        
        n_instances = node_positions.shape[0]
        logging.info(f"[*] Evaluating {dataset_path}")
        
        objs = []
        for i, (node_pos, demand) in enumerate(zip(node_positions, demands)):
            dist_mat = distance_matrix(node_pos, node_pos)
            dist_mat[np.diag_indices_from(dist_mat)] = 1e6 # set diagonal to a large number
            heuristics = scoring_function(dist_mat, demand, CAPACITY) + 1e-6
            aco = ACO(dist_mat, demand, heuristics, CAPACITY, n_ants=N_ANTS)
            obj = aco.run(N_ITERATIONS)
            objs.append(obj.item())
        
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")