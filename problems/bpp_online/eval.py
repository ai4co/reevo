"""
Adapted from https://github.com/google-deepmind/funsearch
"""

import numpy as np
import pickle
import sys

from gpt import priority_v2 as priority


def get_valid_bin_indices(item: float, bins: np.ndarray) -> np.ndarray:
    """Returns indices of bins in which item can fit."""
    return np.nonzero((bins - item) >= 0)[0]


def online_binpack(
    items: tuple[float, ...], bins: np.ndarray
) -> tuple[list[list[float, ...], ...], np.ndarray]:
    """Performs online binpacking of `items` into `bins`."""
    # Track which items are added to each bin.
    packing = [[] for _ in bins]
    # Add items to bins.
    for item in items:
        # Extract bins that have sufficient space to fit item.
        valid_bin_indices = get_valid_bin_indices(item, bins)
        # Score each bin based on heuristic.
        priorities = priority(item, bins[valid_bin_indices])
        # Add item to bin with highest priority.
        best_bin = valid_bin_indices[np.argmax(priorities)]
        bins[best_bin] -= item
        packing[best_bin].append(item)
    # Remove unused bins from packing.
    packing = [bin_items for bin_items in packing if bin_items]
    return packing, bins


def evaluate(instances: dict) -> float:
    """Evaluate heuristic function on a set of online binpacking instances."""
    # List storing number of bins used for each instance.
    num_bins = []
    # Perform online binpacking for each instance.
    for name in instances:
        if name == 'l1_bound':
            continue
        instance = instances[name]
        capacity = instance['capacity']
        items = instance['items']
        items = np.array(items) if isinstance(items, list) else items
        # Create num_items bins so there will always be space for all items,
        # regardless of packing order. Array has shape (num_items,).
        bins = np.array([capacity for _ in range(instance['num_items'])])
        # Pack items into bins and return remaining capacity in bins_packed, which
        # has shape (num_items,).
        _, bins_packed = online_binpack(items.astype(float), bins)
        # If remaining capacity in a bin is equal to initial capacity, then it is
        # unused. Count number of used bins.
        num_bins.append((bins_packed != capacity).sum())
    # Score of heuristic function is negative of average number of bins used
    # across instances (as we want to minimize number of bins).
    return -np.mean(num_bins)


def is_valid_packing(
    packing: list[list[float, ...], ...], items: list[float], capacity: float
) -> bool:
    """Returns whether `packing` is valid.

    Returns whether `packing` is a valid packing of `items` into bins of size
    `capacity`.

    Args:
        packing: Packing of items into bins. List of bins, where each bin contains
        a list of items packed into that bin.
        items: List of item sizes.
        capacity: Capacity of each bin.
    """
    # Check that items in packing are exactly the same as list of input items.
    packed_items = sum(packing, [])  # Join items in each bin into a single list.
    if sorted(packed_items) != sorted(items):
        return False

    # Check that each bin contains less than `capacity` items .
    for bin_items in packing:
        if sum(bin_items) > capacity:
            return False

    return True


if __name__ == "__main__":
    import os
    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    root_dir = sys.argv[2] # reserved for compatibility
    mood = sys.argv[3]
    assert mood in ['train', 'val']
    assert problem_size in [5000, -1]
    
    file_name = f"weibull_5k_{mood}.pickle"
    basepath = os.path.dirname(__file__)
    dataset_path = os.path.join(basepath, "dataset", file_name)
    
    if not os.path.isfile(dataset_path):
        from gen_inst import generate_datasets
        generate_datasets()
    
    dataset = pickle.load(open(dataset_path, 'rb'))
    
    # Evaluate heuristic function on dataset
    avg_num_bins = -evaluate(dataset)
    l1_bound = dataset['l1_bound']
    excess = (avg_num_bins - l1_bound) / l1_bound
    print(file_name)
    print(f'\t Average number of bins: {avg_num_bins}')
    print(f'\t Lower bound on optimum: {l1_bound}')
    print(f'\t Excess: {100 * excess:.2f}%')
    
    print("[*] Average:")
    print(excess * 100)