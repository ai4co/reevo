import hydra
import logging 
import matplotlib.pyplot as plt
import os
from pathlib import Path
import subprocess

from ga import G2A

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.DEBUG)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    
    ga = G2A(cfg, ROOT_DIR)
    best_code_overall, best_desc_overall, best_code_path_overall = ga.evolve()
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Description Overall: {best_desc_overall}")
    logging.info(f"Best Code Path Overall: {best_code_path_overall}")
    
    with open(f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/{cfg.suffix.lower()}.py", 'w') as file:
        file.writelines(best_code_overall + '\n')

    # run test script and redirect stdout to a file "best_code_overall_stdout.txt"
    test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/test.py"
    test_script_stdout = "best_code_overall_stdout.txt"
    logging.info(f"Running test script...: {test_script}")
    with open(test_script_stdout, 'w') as stdout:
        subprocess.run(["python", test_script, ROOT_DIR], stdout=stdout)
    

if __name__ == "__main__":
    main()