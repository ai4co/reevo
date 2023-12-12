import numpy as np

CAPACITY = 50
DEMAND_LOW = 1
DEMAND_HIGH = 9
DEPOT_COOR = [0.5, 0.5]

def gen_instance(n):
    locations = np.random.rand(n, 2)
    demands = np.random.randint(low=DEMAND_LOW, high=DEMAND_HIGH+1, size=n)
    depot = np.array([DEPOT_COOR])
    all_locations = np.concatenate((depot, locations), axis=0)
    all_demands = np.concatenate((np.zeros(1,), demands))
    return np.concatenate((all_demands.reshape(-1, 1), all_locations), axis=1)

if __name__ == "__main__":
    np.random.seed(1234)
    for problem_size in [50]:
        n_instances = 5
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(f'./dataset/val{problem_size}_dataset.npy', dataset)
    
    for problem_size in [20, 50, 100]:
        n_instances = 64
        dataset = []
        for i in range(n_instances):
            inst = gen_instance(problem_size)
            dataset.append(inst)
        dataset = np.array(dataset)
        np.save(f'./dataset/test{problem_size}_dataset.npy', dataset)
