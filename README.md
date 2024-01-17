# ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution

🥳 **Welcome!** This is a codebase that accompanies the paper [*ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution*](). Give ReEvo 5 minutes, and get a state-of-the-art algorithm in return!

TODO: add a diagram

TODO: introduce ReEvo

### Highlights

- TODO: highlights

### Usage

- Set your OpenAI API key as an environment variable `OPENAI_API_KEY`.
- Running logs and intermediate results are saved in `./outputs/main/` by default.
- Datasets are generated on the fly.

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

#### How to apply ReEvo to your problem

TODO


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
- [Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)
- [Algorithm Evolution Using Large Language Model](https://arxiv.org/abs/2311.15249)
- [Mathematical discoveries from program search with large language models](https://github.com/google-deepmind/funsearch)
- [An Example of Evolutionary Computation + Large Language Model Beating Human: Design of Efficient Guided Local Search](https://arxiv.org/abs/2401.02051)
