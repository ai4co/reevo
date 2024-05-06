import os
from .original.eoh import EOH
from .original.getParas import Paras
from .original import prob_rank, pop_greedy
from .problem_adapter import Problem

class EoH:
    def __init__(self, cfg, root_dir) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.problem = Problem(cfg, root_dir)

        self.paras = Paras() 
        self.paras.set_paras(method = "eoh",
                    problem = "tsp_construct",
                    llm_api_endpoint = "api.openai.com",
                    llm_api_key = os.getenv("OPENAI_API_KEY"),
                    llm_model = "gpt-3.5-turbo",
                    ec_pop_size = self.cfg.pop_size,
                    ec_n_pop = 2,
                    exp_output_path = "./",
                    exp_debug_mode = False,
                    eva_timeout=cfg.timeout)
    
    def evolve(self):
        print("- Evolution Start -")

        method = EOH(self.paras, self.problem, prob_rank, pop_greedy)

        results = method.run()

        print("> End of Evolution! ")
        print("----------------------------------------- ")
        print("---     EoH successfully finished !   ---")
        print("-----------------------------------------")

        return results


