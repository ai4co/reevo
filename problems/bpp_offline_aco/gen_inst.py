from typing import NamedTuple
import numpy as np
import numpy.typing as npt

class BPPInstance(NamedTuple):
    n: int
    capacity: int
    demands: npt.NDArray[np.int_]

# Emanuel Falkenauer. A hybrid grouping genetic algorithm for bin packing. Journal of Heuristics,2:5–30, 1996.

DEMAND_LOW = 20
DEMAND_HIGH = 100
CAPACITY = 150
dataset_conf = {
    'train': (500,),
    'val':   (120, 500, 1000),
    'test':  (120, 500, 1000),
}

def generate_dataset(filepath, n, batch_size=64):
    demands = np.random.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=(batch_size, n))
    np.savez(filepath, demands = demands)


def generate_datasets(basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for mood, problem_sizes in dataset_conf.items():
        np.random.seed(len(mood))
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.npz")
            generate_dataset(filepath, n, batch_size=5 if mood =='train' else 64)

def load_dataset(fp) -> list[BPPInstance]:
    data = np.load(fp)
    demands = data['demands']
    instances = []
    n = demands.shape[1]
    for demand in demands:
        instance = BPPInstance(n, CAPACITY, demand)
        instances.append(instance)
    return instances


if __name__ == "__main__":
    generate_datasets()