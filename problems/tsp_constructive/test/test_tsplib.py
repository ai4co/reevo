import numpy as np
from scipy.spatial import distance_matrix
from copy import copy
from tqdm import tqdm

# Datasets and baseline results are drawn from Duflo et al. (2019). A GP Hyper-Heuristic Approach for Generating TSP Heuristics.

names = ["ts225", "rat99", "rl1889", "u1817", "d1655", "bier127", "lin318", "eil51", "d493", "kroB100", "kroC100", "ch130", "pr299", "fl417", "d657", "kroA150", "fl1577", "u724", "pr264", "pr226", "pr439"]

opt = {
    'ts225': 126643,
    'rat99': 1211,
    'rl1889': 316536,
    'u1817': 57201,
    'd1655': 62128,
    'bier127': 118282,
    'lin318': 42029,
    'eil51': 426,
    'd493': 35002,
    'kroB100': 22141,
    'kroC100': 20749,
    'ch130': 6110,
    'pr299': 48191,
    'fl417': 11861,
    'd657': 48912,
    'kroA150': 26524,
    'fl1577': 22249,
    'u724': 41910,
    'pr264': 49135,
    'pr226': 80369,
    'pr439': 107217
 }


def select_next_node_reevo(current_node: int, destination_node: int, unvisited_nodes: set, distance_matrix: np.ndarray) -> int:

        """Select the next node to visit from the unvisited nodes.

        Args:
            current_node: The current node.
            destination_node: The destination node.
            unvisited_nodes: A set of unvisited nodes.
            distance_matrix: A matrix representing the distances between nodes.

        Returns:
            The next node to visit.
        """
        threshold = 0.6
        
        # Define weights for the scoring factors
        weights = {
            'current_distance': 0.25,
            'average_distance_to_unvisited': 0.15,
            'std_dev_distance_to_unvisited': 0.15,
            'destination_distance': 0.1,
            'nearest_neighbor_distance': 0.12,
            'second_nearest_neighbor_distance': 0.08,
            'furthest_neighbor_distance': 0.1,
            'second_furthest_neighbor_distance': 0.07,
        }

        # Normalize the weights
        total_weight = sum(weights.values())
        normalized_weights = {factor: weight / total_weight for factor, weight in weights.items()}

        scores = {}

        for node in unvisited_nodes:
            # Calculate average and standard deviation of distances to unvisited nodes
            distances_to_unvisited = [distance_matrix[node][i] for i in unvisited_nodes if i != node]
            n = len(distances_to_unvisited)
            if n > 0:
                average_distance_to_unvisited = sum(distances_to_unvisited) / (n + 1) # Consider the current node
                std_dev_distance_to_unvisited = np.std(distances_to_unvisited + [distance_matrix[node][current_node]])
            else:
                average_distance_to_unvisited = 0
                std_dev_distance_to_unvisited = 0

            # Calculate the score for the current node
            score = (
                normalized_weights['current_distance'] * distance_matrix[current_node][node]
                - normalized_weights['average_distance_to_unvisited'] * average_distance_to_unvisited
                + normalized_weights['std_dev_distance_to_unvisited'] * std_dev_distance_to_unvisited
                - normalized_weights['destination_distance'] * distance_matrix[destination_node][node]
                - normalized_weights['nearest_neighbor_distance'] * min(distance_matrix[current_node])
                - normalized_weights['second_nearest_neighbor_distance'] * sorted(distance_matrix[current_node])[1]
                - normalized_weights['furthest_neighbor_distance'] * max(distance_matrix[current_node])
                - normalized_weights['second_furthest_neighbor_distance'] * sorted(distance_matrix[current_node])[-2]
            )

            scores[node] = score

        if all(score > threshold for score in scores.values()):
            # Use a greedy strategy to select the node if all scores are above the threshold
            next_node = min(unvisited_nodes, key=lambda node: distance_matrix[current_node][node])
        else:
            # Select the node with the minimum score
            next_node = min(scores, key=scores.get)

        return next_node


def eval_heuristic(node_positions: np.ndarray, start_node: int) -> float:
    problem_size = node_positions.shape[0]
    # calculate distance matrix
    dist_mat = distance_matrix(node_positions, node_positions)
    # set the starting node
    solution = [start_node]
    # init unvisited nodes
    unvisited = set(range(problem_size))
    # remove the starting node
    unvisited.remove(start_node)
    # run the heuristic
    for _ in tqdm(range(problem_size - 1)):
        next_node = select_next_node_reevo(
            current_node=solution[-1],
            destination_node=start_node,
            unvisited_nodes=copy(unvisited),
            distance_matrix=dist_mat.copy(),
        )
        solution.append(next_node)
        if next_node in unvisited:
            unvisited.remove(next_node)
        else:
            raise KeyError(f"Node {next_node} is already visited.")
    
    # calculate the length of the tour
    obj = 0
    for i in range(problem_size):
        obj += dist_mat[solution[i], solution[(i + 1) % problem_size]]
    return obj


if __name__ == '__main__':
    for name in names:
        with open(f'tsplib/{name}.tsp') as f:
            lines = f.readlines()

        # Parse the data
        data = lines[6:-1]
        data = [x.strip().split() for x in data]
        data = [[float(x) for x in row[1:]] for row in data]

        # Scale the data to [0, 1]^2 to align with the training data
        data = np.array(data)
        scale = max(np.max(data, axis=0) - np.min(data, axis=0))
        data = (data - np.min(data, axis=0)) / scale

        # Evaluate the heuristic
        objs = []
        for start_node in range(5):
            obj = eval_heuristic(data, start_node) * scale
            objs.append(obj)
        mean, std = np.mean(objs), np.std(objs)
        print(name)
        print(f"\tObjective: {mean}+-{std}")
        print(f"\tOpt. Gap: {(mean - opt[name]) / opt[name] * 100}%")
        print()
