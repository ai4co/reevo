import math
import numpy as np
import sys
import argparse
import numpy as np
from scipy.spatial import distance_matrix

from gpt import select_next_node

# TSP test
def test_heuristic(node_positions: np.ndarray) -> float:
    '''
    Generate solution for TSP problem using the GPT-generated heuristic algorithm.
    
    Parameters
    ----------
    node_positions : np.ndarray
        2D array of node positions of shape (problem_size, 2).
    
    Returns
    -------
    obj : float
        The length of the generated tour.
    '''
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    start_node = 0
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in range(problem_size - 1):
        next_node = select_next_node(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=unvisited,
            distance_matrix=dist_mat,
        )
        solution.append(next_node)
        unvisited.remove(next_node)
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj
    

if __name__ == '__main__':
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    
    dataset_path = f"{root_dir}/problems/tsp_constructive/dataset/val{problem_size}_dataset.npy"
    node_positions = np.load(dataset_path)
    n_instances = node_positions.shape[0]
    print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
    
    objs = []
    for i in range(n_instances):
        obj = test_heuristic(node_positions[i])
        print(f"[*] Instance {i}: {obj}")
        objs.append(obj)
    
    print("[*] Average:")
    print(np.mean(objs))
   
   
   