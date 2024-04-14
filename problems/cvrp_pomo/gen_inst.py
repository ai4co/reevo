import torch


dataset_conf = {
    'train': (200, 500, 1000),
    'val':   (200, 500, 1000),
    'test':  (200, 500, 1000, 5000),
}


def get_random_problems(batch_size, problem_size):

    depot_xy = torch.rand(size=(batch_size, 1, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 80
    elif problem_size == 500:
        demand_scaler = 100
    elif problem_size == 1000:
        demand_scaler = 250
    elif problem_size == 5000:
        demand_scaler = 500
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    return depot_xy, node_xy, node_demand

def generate_dataset(filepath, n, batch_size=64):
    depot_xy, node_xy, node_demand = get_random_problems(batch_size, n)
    torch.save({
        'depot_xy': depot_xy,
        'node_xy': node_xy,
        'node_demand': node_demand
    }, filepath)

def generate_datasets(basepath = None):
    import os
    basepath = basepath or os.path.join(os.path.dirname(__file__), "dataset")
    os.makedirs(basepath, exist_ok=True)

    for mood, problem_sizes in dataset_conf.items():
        torch.manual_seed(len(mood))
        for n in problem_sizes:
            filepath = os.path.join(basepath, f"{mood}{n}_dataset.pt")
            generate_dataset(filepath, n, batch_size=10 if mood =='train' else 64)

if __name__ == "__main__":
    generate_datasets()