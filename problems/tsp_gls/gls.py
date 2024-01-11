import numpy as np
import numpy.typing as npt
import time
import numba as nb
import concurrent.futures


@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def two_opt_once(distmat, tour, fixed_i = 0):
    '''in-place operation'''
    n = tour.shape[0]
    p = q = 0
    delta = 0
    for i in range(1, n - 1) if fixed_i==0 else range(fixed_i, fixed_i+1):
        for j in range(i + 1, n):
            node_i, node_j = tour[i], tour[j]
            node_prev, node_next = tour[i-1], tour[(j+1) % n]
            if node_prev == node_j or node_next == node_i:
                continue
            change = (  distmat[node_prev, node_j] 
                        + distmat[node_i, node_next]
                        - distmat[node_prev, node_i] 
                        - distmat[node_j, node_next])                    
            if change < delta:
                p, q, delta = i, j, change
    if delta < -1e-6:
        tour[p: q+1] = np.flip(tour[p: q+1])
        return delta
    else:
        return 0.0

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True)
def relocate_once(distmat, tour, fixed_i = 0):
    n = distmat.shape[0]
    delta = p = q = 0
    for i in range(1, n) if fixed_i==0 else range(fixed_i, fixed_i+1):
        node = tour[i]
        prev_node = tour[i-1]
        next_node = tour[(i+1)%n]
        for j in range(n):
            if j == i or j == i-1:
                continue
            prev_insert = tour[j]
            next_insert = tour[(j+1)%n]
            cost = ( - distmat[prev_node, node]
                     - distmat[node, next_node]
                     - distmat[prev_insert, next_insert]
                     + distmat[prev_insert, node]
                     + distmat[node, next_insert]
                     + distmat[prev_node, next_node] )
            if cost < delta:
                delta, p, q = cost, i, j
    if delta >= 0:
        return 0.0
    if p<q:
        tour[p:q+1] = np.roll(tour[p:q+1], -1)
    else:
        tour[q:p+1] = np.roll(tour[q:p+1], 1)
    return delta

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:]), nogil=True)
def calculate_cost(distmat, tour):
    cost = distmat[tour[-1], tour[0]]
    for i in range(len(tour) - 1):
        cost += distmat[tour[i], tour[i+1]]
    return cost

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16, nb.uint16), nogil=True)
def local_search(distmat, cur_tour, fixed_i = 0, count = 1000):
    sum_delta = 0.0
    delta = -1
    while delta < -1e-6 and count > 0:
        delta = 0
        delta += two_opt_once(distmat, cur_tour, fixed_i)
        delta += relocate_once(distmat, cur_tour, fixed_i)
        count -= 1
        sum_delta += delta
    return sum_delta

@nb.njit(nb.void(nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.uint16[:], nb.float32, nb.uint32), nogil=True)
def perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves = 30):
    moves = 0
    n = distmat.shape[0]
    while moves < perturbation_moves:
        # penalize edge
        max_util = 0
        max_util_idx = 0
        for i in range(n-1):
            j = i+1
            u, v = cur_tour[i], cur_tour[j]
            util = guide[u, v] / (1.0 + penalty[u, v])
            if util > max_util:
                max_util_idx, max_util = i, util

        penalty[cur_tour[max_util_idx], cur_tour[max_util_idx+1]] += 1.0
        edge_weight_guided = distmat + k * penalty

        for fixed_i in (max_util_idx, max_util_idx+1):
            if fixed_i == 0 or fixed_i + 1 == n:
                continue
            delta = local_search(edge_weight_guided, cur_tour, fixed_i, 1)
            if delta < 0:
                moves += 1


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.float32[:,:], nb.uint16[:], nb.int32, nb.float32), nogil = True)
def _guided_local_search(
    distmat, guide, init_tour, perturbation_moves = 30, time_limit = 1.0
) -> npt.NDArray[np.uint16]:
    init_cost = calculate_cost(distmat, init_tour)
    k = 0.1 * init_cost / distmat.shape[0]
    penalty = np.zeros_like(distmat)

    with nb.objmode(t_lim=nb.float32):
        t_lim = time.time() + time_limit

    cur_tour = init_tour.copy()
    local_search(distmat, cur_tour, 0, 1000)
    cur_cost = calculate_cost(distmat, cur_tour)
    best_tour, best_cost = cur_tour, cur_cost

    with nb.objmode(now=nb.float32):
        now = time.time()

    while now < t_lim:
        perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)

        local_search(distmat, cur_tour, 0, 1000)
        cur_cost = calculate_cost(distmat, cur_tour)
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour.copy(), cur_cost

        with nb.objmode(now=nb.float32):
            now = time.time()
    return best_tour

def guided_local_search(
    distmat: npt.NDArray[np.float_], 
    guide: npt.NDArray[np.float_], 
    init_tour: npt.NDArray[np.int_], 
    perturbation_moves: int = 30, 
    time_limit: float = 1.0
) -> npt.NDArray[np.uint16]:
    return _guided_local_search(
        distmat = distmat.astype(np.float32),
        guide = guide.astype(np.float32),
        init_tour = init_tour.astype(np.uint16),
        perturbation_moves = perturbation_moves,
        time_limit = time_limit,
    )

def batched_guided_local_search(dist, guide, tours, perturbation_moves = 30, time_limit = 1.0):
    dist = dist.astype(np.float32)
    guide = guide.astype(np.float32)
    tours = tours.astype(np.uint16)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for tour in tours:
            future = executor.submit(_guided_local_search, dist, guide, tour, perturbation_moves = perturbation_moves, time_limit = time_limit)
            futures.append(future)
        return np.stack([f.result() for f in futures])

if __name__ == "__main__":
    import torch
    n = 50
    input = torch.rand(size=(n, 2))
    distances = torch.norm(input[:, None] - input, dim=2, p=2)
    distances[torch.arange(len(distances)), torch.arange(len(distances))] = 1e10
    distances = distances.numpy().astype(np.float32)
    tour = np.arange(n, dtype = np.uint16)
    tours = []
    for i in range(3):
        np.random.shuffle(tour)
        tours.append(tour.copy())
    tours = np.stack(tours)
    print(tours)
    tours = batched_guided_local_search(distances, distances, tours)
    print(tours)