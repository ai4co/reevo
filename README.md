# Large Language Models as Meta-Optimizers


### Usage

- Set your OpenAI API key as an environment variable `OPENAI_API_KEY`.
- Running logs and intermediate results are saved in `./outputs/main/` by default.

#### To run ReEvo
```bash
# e.g., for tsp_aco
python main.py problem=tsp_aco
```

#### To run AEL
```bash
# e.g., for tsp_constructive
python main.py problem=tsp_constructive algorithm=ael mutation_rate=0.2
```

#### Available problems
- Traveling Salesman Problem (TSP): `tsp_aco`, `tsp_aco_black_box`, `tsp_constructive`, `tsp_gls` *(ReEvo only)*
- Capacitated Vehicle Routing Problem (CVRP): `cvrp_aco`, `cvrp_aco_black_box`
- Bin Packing Problem (BPP): `bpp_offline_aco`, `bpp_offline_aco_black_box`, `bpp_online` *(ReEvo only)*
- Multiple Knapsack Problems (MKP): `mkp_aco`, `mkp_aco_black_box`
- Orienteering Problem (OP): `op_aco`, `op_aco_black_box`

### Dependency

- Python >= 3.11
- openai >= 1.0.0
- hydra-core
- scipy

You may install the dependencies above via `pip install -r requirements.txt`.

Problem-specific dependencies:

- `tsp_aco(_black_box)`: pytorch, scikit-learn
- `cvrp_aco(_black_box)` / `mkp_aco(_black_box)` / `op_aco(_black_box)`: pytorch
- `tsp_gls`: numba==0.58

### Acknowledgments
- [DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization](https://github.com/henry-yeh/DeepACO)
- [Mathematical discoveries from program search with large language models](https://github.com/google-deepmind/funsearch)
- [Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)
- [An Example of Evolutionary Computation + Large Language Model Beating Human: Design of Efficient Guided Local Search](https://arxiv.org/abs/2401.02051)
