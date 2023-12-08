import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self, 
                 distances,
                 heuristic,
                 n_ants=30, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu'
                 ):
        
        self.problem_size = len(distances)
        self.distances  = torch.tensor(distances, device=device) if not isinstance(distances, torch.Tensor) else distances
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = torch.tensor(heuristic, device=device) if not isinstance(heuristic, torch.Tensor) else heuristic

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            paths = self.gen_path(require_prob=False)
            costs = self.gen_path_costs(paths)
            
            best_cost, best_idx = costs.min(dim=0)
            if best_cost < self.lowest_cost:
                self.shortest_path = paths[:, best_idx]
                self.lowest_cost = best_cost
            
            self.update_pheronome(paths, costs)

        return self.lowest_cost
       
    @torch.no_grad()
    def update_pheronome(self, paths, costs):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            self.pheromone[path, torch.roll(path, shifts=1)] += 1.0/cost
            self.pheromone[torch.roll(path, shifts=1), path] += 1.0/cost

    @torch.no_grad()
    def gen_path_costs(self, paths):
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
        Returns:
                Lengths of paths: torch tensor with shape (n_ants,)
        '''
        assert paths.shape == (self.problem_size, self.n_ants)
        u = paths.T # shape: (n_ants, problem_size)
        v = torch.roll(u, shifts=1, dims=1)  # shape: (n_ants, problem_size)
        assert (self.distances[u, v] > 0).all()
        return torch.sum(self.distances[u, v], dim=1)

    def gen_path(self, require_prob=False):
        '''
        Tour contruction for all ants
        Returns:
            paths: torch tensor with shape (problem_size, n_ants), paths[:, i] is the constructed tour of the ith ant
            log_probs: torch tensor with shape (problem_size, n_ants), log_probs[i, j] is the log_prob of the ith action of the jth ant
        '''
        start = torch.randint(low=0, high=self.problem_size, size=(self.n_ants,), device=self.device)
        mask = torch.ones(size=(self.n_ants, self.problem_size), device=self.device)
        mask[torch.arange(self.n_ants, device=self.device), start] = 0
        
        paths_list = [] # paths_list[i] is the ith move (tensor) for all ants
        paths_list.append(start)
        
        log_probs_list = [] # log_probs_list[i] is the ith log_prob (tensor) for all ants' actions
        
        prev = start
        for _ in range(self.problem_size-1):
            actions, log_probs = self.pick_move(prev, mask, require_prob)
            paths_list.append(actions)
            if require_prob:
                log_probs_list.append(log_probs)
                mask = mask.clone()
            prev = actions
            mask[torch.arange(self.n_ants, device=self.device), actions] = 0
        
        if require_prob:
            return torch.stack(paths_list), torch.stack(log_probs_list)
        else:
            return torch.stack(paths_list)
        
    def pick_move(self, prev, mask, require_prob):
        '''
        Args:
            prev: tensor with shape (n_ants,), previous nodes for all ants
            mask: bool tensor with shape (n_ants, p_size), masks (0) for the visited cities
        '''
        pheromone = self.pheromone[prev] # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev] # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask) # shape: (n_ants, p_size)
        dist = Categorical(dist)
        actions = dist.sample() # shape: (n_ants,)
        log_probs = dist.log_prob(actions) if require_prob else None # shape: (n_ants,)
        return actions, log_probs