import numpy as np
import numpy.typing as npt
from scipy.spatial import distance_matrix

class TSPInstance:
    def __init__(self, positions: npt.NDArray[np.float_]) -> None:
        self.positions = positions
        self.n = positions.shape[0]
        self.distmat = distance_matrix(positions, positions) + np.eye(self.n)*1e-5
    
dataset_conf = {
    'train': (200,),
    'val':   (20, 50, 100, 200),
    'test':  (20, 50, 100, 200, 500, 1000),
}

def generate_dataset(filepath, n, batch_size=64):
    positions = np.random.random((batch_size, n, 2))
    np.save(filepath, positions)

def generate_datasets(basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for mood, problem_sizes in dataset_conf.items():
        np.random.seed(len(mood))
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.npy")
            generate_dataset(filepath, n, batch_size=10 if mood =='train' else 64)

def load_dataset(fp) -> list[TSPInstance]:
    data = np.load(fp)
    dataset = [TSPInstance(d) for d in data]
    return dataset


if __name__ == "__main__":
    generate_datasets()