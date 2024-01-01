import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self,
                 prize,  # shape [n,]
                 weight, # shape [n, m]
                 heuristic,
                 n_ants=20, 
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 ):

        self.n, self.m = weight.shape
        
        self.prize = prize
        self.weight = weight

        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones(size=(self.n+1, self.n+1), device=device)

        # Fidanova S. Hybrid ant colony optimization algorithm for multiple knapsack problem
        # self.heuristic = (prize / self.weight.sum(dim=1)).unsqueeze(0).repeat(self.n, 1) if heuristic is None else heuristic
        self.heuristic = heuristic
        self.Q = 1/self.prize.sum()

        self.alltime_best_sol = None
        self.alltime_best_obj = 0
        self.device = device
        self.add_dummy_node()
        
    def add_dummy_node(self):
        self.prize = torch.cat((self.prize, torch.tensor([0.], device=self.device))) # (n+1,)
        self.weight = torch.cat((self.weight, torch.zeros((1, self.m), device=self.device)), dim=0) # (n+1, m)
        heu_added_row = torch.cat((self.heuristic, torch.zeros((1, self.n), device=self.device)), dim=0) # (n+1, n)
        self.heuristic = torch.cat((heu_added_row, 1e-10*torch.ones((self.n+1, 1), device=self.device)), dim=1)

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol()         # (n_ants, max_horizon)
            objs = self.gen_sol_obj(sols) # (n_ants,)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs)

        return self.alltime_best_obj, self.alltime_best_sol

    @torch.no_grad()
    def update_pheronome(self, sols, objs):
        self.pheromone = self.pheromone * self.decay 
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += self.Q * obj
        self.pheromone[self.pheromone<1e-10] = 1e-10
        
    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (n_ants, max_horizon)
        Return:
            obj: (n_ants,)
        '''
        return self.prize[solutions.T].sum(dim=1) # (n_ants,)

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        items = torch.randint(low=0, high=self.n, size=(self.n_ants,), device=self.device)
        solutions = [items]
        
        knapsack = torch.zeros(size=(self.n_ants, self.m), device=self.device)  # used capacity
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)

        dummy_mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        dummy_mask[:, -1] = 0
        
        mask, knapsack = self.update_knapsack(mask, knapsack, items)
        dummy_mask = self.update_dummy_state(mask, dummy_mask)
        done = self.check_done(mask)
        while not done:
            items = self.pick_item(items, mask, dummy_mask)
            solutions.append(items)
            mask, knapsack = self.update_knapsack(mask, knapsack, items)
            dummy_mask = self.update_dummy_state(mask, dummy_mask)
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_item(self, items, mask, dummy_mask):
        phe = self.pheromone[items]
        heu = self.heuristic[items]
        dist = ((phe ** self.alpha) * (heu ** self.beta) * mask * dummy_mask) # (n_ants, n+1)
        dist = Categorical(dist)
        item = dist.sample()
        return item
    
    def check_done(self, mask):
        # is mask all zero except for the dummy node?
        return (mask[:, :-1] == 0).all()
    
    def update_dummy_state(self, mask, dummy_mask):
        finished = (mask[: ,:-1] == 0).all(dim=1)
        dummy_mask[finished] = 1
        return dummy_mask
    
    def update_knapsack(self, mask, knapsack, new_item):
        '''
        Args:
            mask: (n_ants, n+1)
            knapsack: (n_ants, m)
            new_item: (n_ants)
        '''
        if new_item is not None:
            mask[torch.arange(self.n_ants), new_item] = 0
            knapsack += self.weight[new_item] # (n_ants, m)
        for ant_idx in range(self.n_ants):
            candidates = torch.nonzero(mask[ant_idx]) # (x, 1)
            if len(candidates) > 1:
                candidates.squeeze_()
                test_knapsack = knapsack[ant_idx].unsqueeze(0).repeat(len(candidates), 1) # (x, m)
                new_knapsack = test_knapsack + self.weight[candidates] # (x, m)
                infeasible_idx = candidates[(new_knapsack > self.n // 2).any(dim=1)]
                mask[ant_idx, infeasible_idx] = 0
        mask[:, -1] = 1
        return mask, knapsack

if __name__ == '__main__':
    from gen_inst import gen_instance
    prize, weight = gen_instance(100, 5) 
    prize = torch.from_numpy(prize)
    weight = torch.from_numpy(weight)
    heu = (prize / weight.sum(dim=1)).unsqueeze(0).repeat(prize.shape[0], 1)
    aco = ACO(prize, weight, heu, n_ants=10)
    for i in range(200):
        obj, sol = aco.run(1)
        print(obj)
        print(sol)
    print(aco.pheromone)