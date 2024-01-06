import torch
from torch import Tensor
import numpy as np
from typing import NamedTuple

class OPInstance(NamedTuple):
    n: int
    coordinate: torch.Tensor
    distance: torch.Tensor
    prize: torch.Tensor
    maxlen: float

def gen_prizes(coordinates: Tensor):
    depot_coor = coordinates[0]
    distances = (coordinates - depot_coor).norm(p=2, dim=-1)
    prizes = 1 + torch.floor(99 * distances / distances.max())
    prizes /= prizes.max()
    return prizes

def gen_distance_matrix(coordinates):
    '''
    Args:
        _coordinates: torch tensor [n_nodes, 2] for node coordinates
    Returns:
        distance_matrix: torch tensor [n_nodes, n_nodes] for EUC distances
    '''
    n_nodes = len(coordinates)
    distances = torch.norm(coordinates[:, None] - coordinates, dim=2, p=2)
    distances[torch.arange(n_nodes), torch.arange(n_nodes)] = 1e9 # note here
    return distances

def get_max_len(n: int) -> float:
    threshold_list = [50, 100, 200, 300]
    maxlen = [3.0,4.0,5.0,6.0]
    for threshold, result in zip(threshold_list, maxlen):
        if n<=threshold:
            return result
    return 7.0

def generate_dataset(filepath, n, batch_size=64):
    coor = np.random.rand(batch_size, n, 2)
    np.savez(filepath, coordinates = coor)

def generate_datasets(basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for mood, seed, problem_sizes in [
        ('train', 1234, (50,)),
        ('val',   3456, (50, 100, 200)),
        ('test',  4567, (50, 100, 200)),
    ]:
        np.random.seed(seed)
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.npz")
            generate_dataset(filepath, n, batch_size=5 if mood =='train' else 64)

def load_dataset(fp) -> list[OPInstance]:
    data = np.load(fp)
    coordinates = data['coordinates']
    instances = []
    n = coordinates[0].shape[0]
    maxlen = get_max_len(n)
    for coord_np in coordinates:
        coord = torch.from_numpy(coord_np)
        distance = gen_distance_matrix(coord)
        prize = gen_prizes(coord)
        instance = OPInstance(n, coord, distance, prize, maxlen)
        instances.append(instance)
    return instances

if __name__ == "__main__":
    generate_datasets()