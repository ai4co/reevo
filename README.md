# Large Language Models as Hyper-Heuristics for Combinatorial Optimization

🥳 **Welcome!** This is a codebase that accompanies the paper *ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution*.

**Give ReEvo 5 minutes, and get a state-of-the-art algorithm in return!**

## Table of Contents

* 1. [ Introduction 🚀](#Introduction)
* 2. [ Exciting Highlights 🌟](#ExcitingHighlights)
* 3. [ Usage 🔑](#Usage)
		* 4.1. [Dependency](#Dependency)
		* 4.2. [To run ReEvo](#TorunReEvo)
		* 4.3. [Available problems](#Availableproblems)
		* 4.4. [Simple steps to apply ReEvo to your problem](#SimplestepstoapplyReEvotoyourproblem)



##  1. <a name='Introduction'></a> Introduction 🚀

![Diagram of ReEvo](./assets/reevo.jpg)

We introduce **Language Hyper-Heuristics (LHHs)**, an emerging variant of Hyper-Heuristics (HHs) that leverages LLMs for heuristic generation, featuring **minimal manual intervention and open-ended heuristic spaces**.

To empower LHHs, we present **Reflective Evolution (ReEvo)**, a generic searching framework that emulates the reflective design approach of human experts while much surpassing human capabilities with its scalable LLM inference, Internet-scale domain knowledge, and powerful evolutionary search.


##  2. <a name='ExcitingHighlights'></a> Exciting Highlights 🌟

We can improve the following types of algorithms:
- Neural Combinatorial Optimization (NCO)
- Genetic Algorithm (GA)
- Ant Colony Optimization (ACO)
- Guided Local Search (GLS)
- Constructive Heuristics

on the following problems:
- Traveling Salesman Problem (TSP)
- Capacitated Vehicle Routing Problem (CVRP)
- Orienteering Problem (OP)
- Multiple Knapsack Problems (MKP)
- Bin Packing Problem (BPP)
- Decap Placement Problem (DPP)

with both black-box and white-box settings.

##  3. <a name='Usage'></a> Usage 🔑

- Set your LLM API key (OpenAI API, ZhiPu API, Llama API) [here](https://github.com/ai4co/LLM-as-HH/blob/5fa30b9da3ecb80b8a658352d26df08893f88a6c/utils/utils.py#L9-L27) or as an environment variable.
- Running logs and intermediate results are saved in `./outputs/main/` by default.
- Datasets are generated on the fly.
- Some test notebooks are provided in `./problems/*/test.ipynb`.

####  3.1. <a name='Dependency'></a>Dependency

- Python >= 3.11
- openai >= 1.0.0
- hydra-core
- scipy

You may install the dependencies above via `pip install -r requirements.txt`.

Problem-specific dependencies:

- `tsp_aco(_black_box)`: pytorch, scikit-learn
- `cvrp_aco(_black_box)` / `mkp_aco(_black_box)` / `op_aco(_black_box)` / `NCO`: pytorch
- `tsp_gls`: numba==0.58


####  3.2. <a name='TorunReEvo'></a>To run ReEvo
```bash
# e.g., for tsp_aco
python main.py problem=tsp_aco
```
Check out `./cfg/` for more options.

####  3.3. <a name='Availableproblems'></a>Available problems
- Traveling Salesman Problem (TSP): `tsp_aco`, `tsp_aco_black_box`, `tsp_constructive`, `tsp_gls`, `tsp_pomo`, `tsp_lehd`
- Capacitated Vehicle Routing Problem (CVRP): `cvrp_aco`, `cvrp_aco_black_box`, `cvrp_pomo`, `cvrp_lehd`
- Bin Packing Problem (BPP): `bpp_offline_aco`, `bpp_offline_aco_black_box`, `bpp_online`
- Multiple Knapsack Problems (MKP): `mkp_aco`, `mkp_aco_black_box`
- Orienteering Problem (OP): `op_aco`, `op_aco_black_box`
- Decap Placement Problem (DPP): `dpp_ga`

####  3.4. <a name='SimplestepstoapplyReEvotoyourproblem'></a>Simple steps to apply ReEvo to your problem

- Define your problem in `./cfg/problem/`.
- Generate problem instances and implement the evaluation pipeline in `./problems/`.
- Add function_description, function_signature, and seed_function in `./prompts/`.
