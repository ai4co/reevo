import os
import numpy as np

def generate_datasets():
    basepath = os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    np.random.seed(1234)

    for problem_size in [50]:
        n_instances = 5
        test_dataset = np.random.rand(n_instances, problem_size, 2)
        np.save(os.path.join(basepath, f'train{problem_size}_dataset.npy'), test_dataset)

    for problem_size in [20, 50, 100]:
        n_instances = 64
        test_dataset = np.random.rand(n_instances, problem_size, 2)
        np.save(os.path.join(basepath, f'val{problem_size}_dataset.npy'), test_dataset)
        
    for problem_size in [20, 50, 100]:
        n_instances = 64
        test_dataset = np.random.rand(n_instances, problem_size, 2)
        np.save(os.path.join(basepath, f'test{problem_size}_dataset.npy'), test_dataset)

if __name__ == "__main__":
    generate_datasets()