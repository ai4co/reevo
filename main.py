import hydra
import logging 
import matplotlib.pyplot as plt
import os
from pathlib import Path

from ga import GA_LLM

ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.DEBUG)

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    
    ga = GA_LLM(cfg, ROOT_DIR)
    ga.evolve()
    

if __name__ == "__main__":
    main()