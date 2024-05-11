from aco import ACO
import numpy as np
import logging
from gen_inst import BPPInstance, load_dataset, dataset_conf
import sys
sys.path.insert(0, "../../../")

import gpt
from utils.utils import get_heuristic_name


possible_func_names = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

heuristic_name = get_heuristic_name(gpt, possible_func_names)
heuristics = getattr(gpt, heuristic_name)


N_ITERATIONS = 15
N_ANTS = 20
SAMPLE_COUNT = 200

def solve(inst: BPPInstance, mode = 'sample'):
    heu = heuristics(inst.demands.copy(), inst.capacity) # normalized in ACO
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf
    aco = ACO(inst.demands, heu.astype(float), capacity = inst.capacity, n_ants=N_ANTS, greedy=False)
    if mode == 'sample':
        obj, _ = aco.sample_only(SAMPLE_COUNT)
    else:
        obj, _ = aco.run(N_ITERATIONS)
    return obj

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2]
    mood = sys.argv[3]
    method = sys.argv[4] if len(sys.argv) >= 5 else 'aco'
    assert mood in ['train', 'val']
    assert method in ['sample', 'aco']

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npz")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
        dataset = load_dataset(dataset_path)
        n_instances = len(dataset)

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")
        
        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance, mode=method)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)
        
        print("[*] Average:")
        print(np.mean(objs))

    else: # mood == 'val'
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.npz")
            dataset = load_dataset(dataset_path)
            n_instances = dataset[0].n
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(dataset):
                obj = solve(instance, mode=method)
                objs.append(obj)
            
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")