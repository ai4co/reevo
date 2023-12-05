import hydra
import numpy as np 
import json
import logging 
import matplotlib.pyplot as plt
import os
from openai import OpenAI
import re
import subprocess
from pathlib import Path
import shutil
import time 
from pprint import pprint

from utils.utils import * 
from ga import GA_LLM

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ROOT_DIR = os.getcwd()

@hydra.main(config_path="cfg", config_name="config", version_base="1.1")
def main(cfg):
    workspace_dir = Path.cwd()
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.model}")
    
    ga = GA_LLM(cfg, ROOT_DIR)
    

if __name__ == "__main__":
    main()