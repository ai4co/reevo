import numpy as np
import numpy.typing as npt
import numba as nb
import concurrent.futures

FloatArray = npt.NDArray[np.float_]
IntArray = npt.NDArray[np.int_]
usecache = True

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True, cache = usecache)
def _two_opt_once(distmat, tour, fixed_i = 0):
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

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16), nogil=True, cache = usecache)
def _relocate_once(distmat, tour, fixed_i = 0):
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

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:]), nogil=True, cache = usecache)
def _calculate_cost(distmat, tour):
    cost = distmat[tour[-1], tour[0]]
    for i in range(len(tour) - 1):
        cost += distmat[tour[i], tour[i+1]]
    return cost

@nb.njit(nb.float32(nb.float32[:,:], nb.uint16[:], nb.uint16, nb.uint16), nogil=True, cache = usecache)
def _local_search(distmat, cur_tour, fixed_i = 0, count = 1000):
    sum_delta = 0.0
    delta = -1
    while delta < 0 and count > 0:
        delta = 0
        delta += _two_opt_once(distmat, cur_tour, fixed_i)
        delta += _relocate_once(distmat, cur_tour, fixed_i)
        count -= 1
        sum_delta += delta
    return sum_delta

@nb.njit(nb.void(nb.float32[:,:], nb.float32[:,:], nb.float32[:,:], nb.uint16[:], nb.float32, nb.uint32), nogil=True, cache = usecache)
def _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves = 30):
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
            delta = _local_search(edge_weight_guided, cur_tour, fixed_i, 1)
            if delta < 0:
                moves += 1

@nb.njit(nb.uint16[:](nb.float32[:,:], nb.uint16), nogil=True, cache = usecache)
def _init_nearest_neighbor(distmat, start):
    n = distmat.shape[0]
    tour = np.zeros(n, dtype=np.uint16)
    visited = np.zeros(n, dtype=np.bool_)
    visited[start] = True
    tour[0] = start
    for i in range(1, n):
        min_dist = np.inf
        min_idx = -1
        for j in range(n):
            if not visited[j] and distmat[tour[i-1], j] < min_dist:
                min_dist = distmat[tour[i-1], j]
                min_idx = j
        tour[i] = min_idx
        visited[min_idx] = True
    return tour


@nb.njit(nb.uint16[:](nb.float32[:,:], nb.float32[:,:], nb.uint16, nb.int32, nb.uint16), nogil = True, cache = usecache)
def _guided_local_search(
    distmat, guide, start, perturbation_moves = 30, iter_limit = 1000
) -> npt.NDArray[np.uint16]:
    penalty = np.zeros_like(distmat)

    best_tour = _init_nearest_neighbor(distmat, start)
    _local_search(distmat, best_tour, 0, 1000)
    best_cost = _calculate_cost(distmat, best_tour)
    k = 0.1 * best_cost / distmat.shape[0]
    cur_tour = best_tour.copy()
    
    for _ in range(iter_limit):
        _perturbation(distmat, guide, penalty, cur_tour, k, perturbation_moves)
        _local_search(distmat, cur_tour, 0, 1000)
        cur_cost = _calculate_cost(distmat, cur_tour)
        if cur_cost < best_cost:
            best_tour, best_cost = cur_tour.copy(), cur_cost
    return best_tour

def guided_local_search(
    distmat: FloatArray, 
    guide: FloatArray, 
    perturbation_moves: int = 30, 
    iter_limit: int = 1000
) -> npt.NDArray[np.uint16]:
    return _guided_local_search(
        distmat = distmat.astype(np.float32),
        guide = guide.astype(np.float32),
        start = 0,
        perturbation_moves = perturbation_moves,
        iter_limit = iter_limit,
    )

def multi_start_guided_local_search(
    dist: FloatArray, 
    guide: FloatArray, 
    n_starts: int = 10,
    perturbation_moves = 30, 
    iter_limit = 1000
):
    dist = dist.astype(np.float32)
    guide = guide.astype(np.float32)
    start_nodes = np.arange(n_starts).astype(np.uint16)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for start in start_nodes:
            future = executor.submit(_guided_local_search, dist, guide, start, perturbation_moves = perturbation_moves, iter_limit = iter_limit)
            futures.append(future)
        tours = [f.result() for f in futures]
        # Calculate costs and return the best tour
        costs = np.array([_calculate_cost(dist, tour) for tour in tours])
        best_tour = tours[np.argmin(costs)]
        return best_tour