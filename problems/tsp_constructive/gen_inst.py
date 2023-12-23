import numpy as np

np.random.seed(1234)

for problem_size in [50]:
    n_instances = 64
    test_dataset = np.random.rand(n_instances, problem_size, 2)
    np.save(f'./dataset/train{problem_size}_dataset.npy', test_dataset)

for problem_size in [50]:
    n_instances = 64
    test_dataset = np.random.rand(n_instances, problem_size, 2)
    np.save(f'./dataset/val{problem_size}_dataset.npy', test_dataset)
    
for problem_size in [20, 50, 100, 200, 500, 1000]:
    n_instances = 64
    test_dataset = np.random.rand(n_instances, problem_size, 2)
    np.save(f'./dataset/test{problem_size}_dataset.npy', test_dataset)