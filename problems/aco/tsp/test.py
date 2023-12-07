from aco import ACO
import sys
import numpy as np
from scipy.spatial import distance_matrix

from gpt import scoring_function


N_ITERATIONS = 20
N_ANTS = 20


if __name__ == "__main__":
    root_dir = sys.argv[1]
    for problem_size in [20, 50, 100, 200, 500, 1000]:
        dataset_path = f"{root_dir}/problems/tsp_constructive/dataset/test{problem_size}_dataset.npy"
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        objs = []
        for i, node_pos in enumerate(node_positions):
            dist_mat = distance_matrix(node_pos, node_pos)
            heuristics = scoring_function(dist_mat)
            aco = ACO(dist_mat, heuristics, n_ants=N_ANTS)
            obj = aco.run(N_ITERATIONS)
            objs.append(obj)
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")