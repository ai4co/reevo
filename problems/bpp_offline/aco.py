import numpy as np
import numpy.typing as npt
from typing import Tuple, List, Optional, Annotated

IntArray = npt.NDArray[np.int_]
FloatArray = npt.NDArray[np.float_]

CAPACITY = 150

def organize_path(path: IntArray) -> Tuple[int, IntArray]:
    order = {}
    result = np.zeros_like(path)
    for i, v in enumerate(path):
        if v in order:
            result[i] = order[v]
        else:
            result[i] = order[v] = len(order)
    return len(order), result

def calculate_path_cost_fitness(vacancies: IntArray, capacity: int) -> Tuple[int, float]:
    occupied = (capacity - vacancies[vacancies!=capacity]).astype(float)
    cost = len(occupied)
    result = ((occupied/capacity)**2).sum().item()/cost
    return cost, result

class ACO(object):
    def __init__(self,  # 0: depot
                 demand: IntArray,   # (n, )
                 heuristic: FloatArray,
                 max_bin_count: Optional[int]= None,
                 capacity=CAPACITY,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 ):
        
        self.problem_size = len(demand)
        self.capacity = capacity
        self.demand = demand
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.max_bin_count = max_bin_count or self.problem_size
        
        self.pheromone: FloatArray = np.ones((self.problem_size, self.max_bin_count))*0.01 # problem_size x bin_count
        heuristic = heuristic/heuristic.max() # normalize
        heuristic[heuristic<1e-5] = 1e-5
        self.heuristic: FloatArray = heuristic # problem_size x bin_count

        self.shortest_path: IntArray = np.array([0])
        self.best_cost = self.max_bin_count
        self.best_fitness = 0.0

        self._ordinal = np.arange(self.problem_size, dtype=int) # for indexing
        self.greedy_mode = False
    
    def run(self, iterations: int) -> Tuple[int, IntArray]:
        self.greedy_mode = False
        for _ in range(iterations):
            prob = (self.pheromone**self.alpha)*(self.heuristic**self.beta) # problem_size x bin_count
            paths, costs, fitnesses = self.gen_paths(self.n_ants, prob)
            best_index = costs.argmin()
            best_cost = costs[best_index].item()
            if best_cost < self.best_cost:
                self.shortest_path = paths[best_index]
                self.best_cost = best_cost
            self.update_pheronome(paths, fitnesses)
        return organize_path(self.shortest_path)

    def sample_only(self, count: int) -> Tuple[int, IntArray]:
        self.greedy_mode = True
        paths, costs, _ = self.gen_paths(count, self.heuristic)
        best_index = costs.argmin()
        best_path = paths[best_index]
        return organize_path(best_path)

    def update_pheronome(self, paths: List[IntArray], fitnesses: FloatArray):
        delta_phe = np.zeros_like(self.pheromone) # problem_size x bin_count
        for p, f in zip(paths, fitnesses):
            delta_phe[self._ordinal, p] = f / self.n_ants
        self.pheromone *= self.decay
        self.pheromone += delta_phe

    def gen_paths(self, count: int, prob_matrix: FloatArray) -> Tuple[List[IntArray], IntArray, FloatArray]:
        paths, costs, fitnesses = [], [], []
        for _ in range(count):
            path, cost, fitness = self.sample_path(prob_matrix)
            if path is not None:
                paths.append(path)
                costs.append(cost)
                fitnesses.append(fitness)
        assert len(paths) > 0, "No valid path is generated, please consider increase max_bin_count"
        return paths, np.array(costs, dtype=int), np.array(fitnesses, dtype=float)
    
    def sample_path(self, prob_matrix: FloatArray
                    ) -> Tuple[
                        Annotated[IntArray | None, "sampled path, None for invalid path"], 
                        Annotated[int, "used bins"], 
                        Annotated[float, "fitness"]]:
        vacancy = np.ones(self.max_bin_count, dtype=int)*self.capacity
        path = np.zeros(self.problem_size, dtype=int) # x=path[i] => put item i in bin x
        order = self._ordinal.copy()
        np.random.shuffle(order)
        for index in order:
            demand = self.demand[index]
            mask = demand <= vacancy
            if np.count_nonzero(mask) == 0:
                return None, 0, 0.0
            if self.greedy_mode:
                prob = prob_matrix[index]*mask
                sampled = prob.argmax()
            else:
                prob = (prob_matrix[index]+1e-6)*mask
                sampled = np.random.choice(self.max_bin_count, p=prob/prob.sum())
            path[index] = sampled
            vacancy[sampled] -= demand
        cost, fitness = calculate_path_cost_fitness(vacancy, self.capacity)
        return path, cost, fitness

    
    def is_valid_path(self, path: IntArray) -> bool:
        # not used
        if path.shape[0] != self.problem_size:
            return False
        bins, path = organize_path(path)
        occupied = np.zeros(bins, dtype=int)
        for i, v in enumerate(path):
            occupied[v] += self.demand[i]
            if occupied[v] > self.capacity:
                return False
        return True

if __name__ == '__main__':
    np.random.seed(1)
    demand = np.random.randint(3, 15, size=(60,))
    np.random.seed()
    print(*demand)
    aco = ACO(demand, np.random.rand(60, 40), capacity=40, max_bin_count=40)
    print(*aco.run(50))

