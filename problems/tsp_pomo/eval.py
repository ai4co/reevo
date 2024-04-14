##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = False
CUDA_DEVICE_NUM = 0


##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# sys.path.insert(0, "..")  # for problem_def
# sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils import create_logger, copy_all_src

from TSPTester import TSPTester as Tester

from gen_inst import dataset_conf

##########################################################################################
# parameters

env_params = {
    'problem_size': 100,
    'pomo_size': 1,
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './checkpoints',  # directory path of pre-trained model and log files saved.
        'epoch': 3100,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size': 10,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 100,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']

logger_params = {
    'log_file': {
        'desc': 'test__tsp100_longTrain',
        'filename': 'log.txt'
    }
}

##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    avg_aug_obj = tester.run()
    return avg_aug_obj


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 100


##########################################################################################

if __name__ == "__main__":
    import sys
    import os

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]
    
    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.pt")):
        from gen_inst import generate_datasets, dataset_conf
        generate_datasets()

    if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-3100.pt")):
        raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")

    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
        env_params['test_file_path'] = dataset_path
        env_params['problem_size'] = problem_size
        tester_params['test_episodes'] = 10
        tester_params['test_batch_size'] = 10
        avg_obj = main()
        print("[*] Average:")
        print(avg_obj)
    
    else:
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
            env_params['test_file_path'] = dataset_path
            env_params['problem_size'] = problem_size
            tester_params['test_episodes'] = 64
            tester_params['test_batch_size'] = 64
            avg_obj = main()
            print(f"[*] Average for {problem_size}: {avg_obj}")