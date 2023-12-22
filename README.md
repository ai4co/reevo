# Large Language Models as Meta-Optimizers


### Usage

Please generate datasets before running. Refer to `problems/*/dataset/readme.md` for details.


```bash
# for tsp_constructive
python main.py problem=tsp_constructive

# for tsp_aco
python main.py problem=tsp_aco

# for cvrp_aco
python main.py problem=cvrp_aco

# for online BPP
python main.py problem=bpp_online

# when using GPT-4 Turbo, add:
model=gpt-4-1106-preview
```


### Dependency

- Python >= 3.9
- openai >= 1.0.0
- hydra-core

### Acknowledgments
- [DeepACO: Neural-enhanced Ant Systems for Combinatorial Optimization](https://github.com/henry-yeh/DeepACO)
- [Mathematical discoveries from program search with large language models](https://github.com/google-deepmind/funsearch)
- [Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)