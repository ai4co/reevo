# Large Language Models as Meta-Optimizers


### Usage

Please generate datasets before running. Refer to `problems/*/dataset/readme.md` for details.


```bash
# for tsp_constructive
python main.py problem=tsp_constructive problem_type=constructive diversify=True
python main.py problem=tsp_constructive problem_type=constructive model=gpt-4-1106-preview diversify=False # using GPT-4-turbo

# for tsp_aco
python main.py problem=tsp_aco problem_type=aco

# for cvrp_aco
python main.py problem=cvrp_aco problem_type=aco
```


### Dependency

- Python >= 3.9
- openai >= 1.0.0
- hydra-core

### Acknowledgments

[Eureka: Human-Level Reward Design via Coding Large Language Models](https://github.com/eureka-research/Eureka)