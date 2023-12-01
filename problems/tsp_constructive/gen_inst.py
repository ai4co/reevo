import numpy as np

# set the random seed
np.random.seed(0)

# generate ans save test instances for TSP
for problem_size in [50]:
    n_instances = 64
    test_dataset = np.random.rand(n_instances, problem_size, 2)
    np.save(f'./dataset/val{problem_size}_dataset.npy', test_dataset)