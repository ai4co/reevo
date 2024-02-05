# ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution

🥳 **Welcome!** This is a codebase that accompanies the paper [*ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution*](https://arxiv.org/abs/2402.01145).

**Give ReEvo 5 minutes, and get a state-of-the-art algorithm in return!**


## 🚀 Introduction

![Diagram of ReEvo](./assets/reevo.png)

We introduce **Language Hyper-Heuristics (LHHs)**, an emerging variant of Hyper-Heuristics (HHs) that leverages LLMs for heuristic generation, featured by **minimal manual intervention and open-ended heuristic spaces**.

To empower LHHs, we present **Reflective Evolution (ReEvo)**, a generic searching framework that emulates the reflective design approach of human experts while much surpassing human capabilities with its scalable LLM inference, Internet-scale domain knowledge, and powerful evolutionary search.


## 🌟 Exciting Highlights!

- State-of-the-art Guided Local Search (GLS) delivered by 20 minutes of ReEvo. We present an efficient Python implementation of GLS using Numba (`./problems/tsp_gls/gls.py`).
- Better Ant Colony Optimization (ACO) heuristics than [DeepACO](https://github.com/henry-yeh/DeepACO) and human designs.
- We include 13 Combinatorial Optimization Problem settings in this repo.
- Emergent capabilities to evolve heuristics under black-box prompting.
- Maybe we no longer need to predefine a heuristic space for future HH research.
- Maybe we can beat Neural Combinatorial Optimization solvers using ten lines of ReEvo-generated heuristics.
- Maybe we can interpret and verbalize genetic cues in Evolutionary Algorithms (EAs) with LLMs.


## 🔑 Usage

- Set your OpenAI API key as an environment variable `OPENAI_API_KEY`.
- Running logs and intermediate results are saved in `./outputs/main/` by default.
- Datasets are generated on the fly.
- Some test notebooks are provided in `./problems/*/test.ipynb`.

#### Dependency

- Python >= 3.11
- openai >= 1.0.0
- hydra-core
- scipy

You may install the dependencies above via `pip install -r requirements.txt`.

Problem-specific dependencies:

- `tsp_aco(_black_box)`: pytorch, scikit-learn
- `cvrp_aco(_black_box)` / `mkp_aco(_black_box)` / `op_aco(_black_box)`: pytorch
- `tsp_gls`: numba==0.58


#### To run ReEvo
```bash
# e.g., for tsp_aco
python main.py problem=tsp_aco
```
Check out `./cfg/` for more options.

You can try a baseline LHH [AEL](https://arxiv.org/abs/2311.15249) by setting `algorithm=ael`. E.g.,
```bash
python main.py problem=tsp_aco algorithm=ael mutation_rate=0.2
```

#### Available problems
- Traveling Salesman Problem (TSP): `tsp_aco`, `tsp_aco_black_box`, `tsp_constructive`, `tsp_gls` *(ReEvo only)*
- Capacitated Vehicle Routing Problem (CVRP): `cvrp_aco`, `cvrp_aco_black_box`
- Bin Packing Problem (BPP): `bpp_offline_aco`, `bpp_offline_aco_black_box`, `bpp_online` *(ReEvo only)*
- Multiple Knapsack Problems (MKP): `mkp_aco`, `mkp_aco_black_box`
- Orienteering Problem (OP): `op_aco`, `op_aco_black_box`

#### Simple steps to apply ReEvo to your problem

- Define your problem in `./cfg/problem/`.
- Generate problem instances and implement the evaluation pipeline in `./problems/`.
- Add function_description, function_signature, and seed_function in `./prompts/`.


## 🤩 Citation

If you encounter any difficulty using our code, please do not hesitate to submit an issue or directly contact us! If you find our work helpful (or if you would be so kind as to offer us some encouragement), please consider kindly giving us a star, and citing our paper.

```bibtex
@misc{ye2024reevo,
      title={ReEvo: Large Language Models as Hyper-Heuristics with Reflective Evolution}, 
      author={Haoran Ye and Jiarui Wang and Zhiguang Cao and Guojie Song},
      year={2024},
      eprint={2402.01145},
      archivePrefix={arXiv},
      primaryClass={cs.NE}
}
```

## 🫡 Acknowledgments
We are very grateful to [Federico Berto](https://github.com/fedebotu), [Yuan Jiang](https://github.com/jiang-yuan), [Yining Ma](https://github.com/yining043), [Chuanbo Hua](https://github.com/cbhua), and [AI4CO community](https://github.com/ai4co) for valuable discussions and feedback.

Also, our work is built upon the following projects, among others:
- [DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization](https://github.com/henry-yeh/DeepACO)
- [Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)
- [Algorithm Evolution Using Large Language Model](https://arxiv.org/abs/2311.15249)
- [Mathematical discoveries from program search with large language models](https://github.com/google-deepmind/funsearch)
- [An Example of Evolutionary Computation + Large Language Model Beating Human: Design of Efficient Guided Local Search](https://arxiv.org/abs/2401.02051)
