import torch

dataset_conf = {
    'train': (200, 500, 1000),
    'val':   (200, 500, 1000),
    'test':  (200, 500, 1000),
}

def generate_dataset(filepath, n, batch_size=64):
    positions = torch.rand(batch_size, n, 2)
    torch.save(positions, filepath)

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