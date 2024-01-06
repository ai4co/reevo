import torch
from torch.distributions import Categorical

class ACO():

    def __init__(self,
                 prizes,
                 distances,
                 max_len,
                 heuristic,
                 n_ants=20,
                 decay=0.9,
                 alpha=1,
                 beta=1,
                 device='cpu',
                 ):
        
        self.n = len(prizes)
        self.distances = distances
        self.prizes = prizes
        self.max_len = max_len
        
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        
        self.pheromone = torch.ones_like(self.distances)
        self.heuristic = heuristic
        
        self.Q = 1 / prizes.sum()
        
        self.alltime_best_sol = None
        self.alltime_best_obj = 0

        self.device = device
        
        self.add_dummy_node()
        
    def add_dummy_node(self):
        '''
        One has to sparsify the graph first before adding dummy node
        distance: 
                [[1e9 , x   , x   , 0  ],
                [x   , 1e9 , x   , 0  ],
                [x   , x   , 1e9 , 0  ],
                [1e10, 1e10, 1e10, 0  ]]
        pheromone: [1]
        heuristic: [>0]
        prizes: [x,x,...,0]
        '''
        self.prizes = torch.cat((self.prizes, torch.tensor([1e-10], device=self.device)))
        distances = torch.cat((self.distances, 1e10 * torch.ones(size=(1, self.n), device=self.device)), dim=0)
        self.distances = torch.cat((distances, 1e-10 + torch.zeros(size=(self.n+1, 1), device=self.device)), dim=1)

        self.heuristic = torch.cat((self.heuristic, torch.zeros(size=(1, self.n), device=self.device)), dim=0) # cannot reach other nodes from dummy node
        self.heuristic = torch.cat((self.heuristic, torch.ones(size=(self.n+1, 1), device=self.device)), dim=1)

        self.pheromone = torch.ones_like(self.distances)
        self.distances[self.distances == 1e-10] = 0
        self.prizes[-1] = 0

    @torch.no_grad()
    def run(self, n_iterations):
        for _ in range(n_iterations):
            sols = self.gen_sol()
            objs = self.gen_sol_obj(sols)
            sols = sols.T
            best_obj, best_idx = objs.max(dim=0)
            if best_obj > self.alltime_best_obj:
                self.alltime_best_obj = best_obj
                self.alltime_best_sol = sols[best_idx]
            self.update_pheronome(sols, objs, best_obj, best_idx)
        return self.alltime_best_obj, self.alltime_best_sol
       
    
    @torch.no_grad()
    def update_pheronome(self, sols, objs, best_obj, best_idx):
        self.pheromone = self.pheromone * self.decay
        for i in range(self.n_ants):
            sol = sols[i]
            obj = objs[i]
            self.pheromone[sol[:-1], torch.roll(sol, shifts=-1)[:-1]] += self.Q * obj
                
    
    @torch.no_grad()
    def gen_sol_obj(self, solutions):
        '''
        Args:
            solutions: (max_len, n_ants)
        '''
        objs = self.prizes[solutions.T].sum(dim=1)
        return objs

    def gen_sol(self):
        '''
        Solution contruction for all ants
        '''
        solutions = []

        solutions = [torch.zeros(size=(self.n_ants,), device=self.device, dtype=torch.int64)]
        mask = torch.ones(size=(self.n_ants, self.n+1), device=self.device)
        done = torch.zeros(size=(self.n_ants,), device=self.device)
        travel_dis = torch.zeros(size=(self.n_ants,), device=self.device)
        cur_node = torch.zeros(size=(self.n_ants,), dtype=torch.int64, device=self.device)
        
        mask = self.update_mask(travel_dis, cur_node, mask)
        done = self.check_done(mask)
        # construction
        while not done:
            nxt_node = self.pick_node(mask, cur_node) # pick action
            # update solution and log_probs
            solutions.append(nxt_node) 
            # update travel_dis, cur_node and mask
            travel_dis += self.distances[cur_node, nxt_node]
            cur_node = nxt_node
            mask = self.update_mask(travel_dis, cur_node, mask)
            # check done
            done = self.check_done(mask)
        return torch.stack(solutions)
    
    def pick_node(self, mask, cur_node):
        pheromone = self.pheromone[cur_node] # shape: (n_ants, p_size+1)
        heuristic = self.heuristic[cur_node] # shape: (n_ants, p_size+1)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * mask)
        dist = Categorical(dist)
        item = dist.sample()
        return item  # (n_ants,)
    
    def update_mask(self, travel_dis, cur_node, mask):
        '''
        Args:
            travel_dis: (n_ants,)
            cur_node: (n_ants,)
            mask: (n_ants, n+1)
        '''
        mask[torch.arange(self.n_ants), cur_node] = 0

        for ant_id in range(self.n_ants):
            if cur_node[ant_id] != self.n: # if not at dummy node
                _mask = mask[ant_id]
                candidates = torch.nonzero(_mask).squeeze()
                # after going to candidate node from cur_node, can it return to depot?
                trails = travel_dis[ant_id] + self.distances[cur_node[ant_id], candidates] + self.distances[candidates, 0]
                fail_idx = candidates[trails > self.max_len]
                _mask[fail_idx] = 0
                
        mask[:, -1] = 0 # mask the dummy node for all ants
        go2dummy = (mask[:, :-1] == 0).all(dim=1) # unmask the dummy node for these ants
        mask[go2dummy, -1] = 1
        return mask
    
    def check_done(self, mask):
        # is all masked ?
        return (mask[:, :-1] == 0).all()
        

if __name__ == '__main__':
    import time
    torch.set_printoptions(precision=4,sci_mode=False)
    from gen_inst import gen_prizes, gen_distance_matrix
    device = 'cpu'
    coor = torch.rand(size=(100, 2), device=device)
    prizes, max_len = gen_prizes(coor), 9
    distances = gen_distance_matrix(coor)
    aco = ACO(prizes,distances,max_len,None)
    start = time.time()
    for i in range(100):
        obj, sol = aco.run(1)
        print(obj)
    print(time.time() - start)