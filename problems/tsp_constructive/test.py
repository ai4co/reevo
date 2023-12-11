import math
import numpy as np
import sys
import logging
import argparse
import numpy as np
from scipy.spatial import distance_matrix

from gpt import select_next_node

# TSP evaluation
def eval_heuristic(node_positions: np.ndarray) -> float:
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
        if next_node in unvisited:
            unvisited.remove(next_node)
        elif len(unvisited) + len(solution) ==  problem_size:
            pass
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj
    

if __name__ == '__main__':
    root_dir = sys.argv[1]
    for problem_size in [20, 50, 100, 200, 500, 1000]:
        dataset_path = f"{root_dir}/problems/tsp_constructive/dataset/test{problem_size}_dataset.npy"
        logging.info(f"[*] Evaluating {dataset_path}")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        objs = []
        for i in range(n_instances):
            obj = eval_heuristic(node_positions[i])
            objs.append(obj)
        print(f"[*] Average for {problem_size}: {np.mean(objs)}")