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

from CVRPTester import CVRPTester as Tester


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
        'path': './checkpoints/',  # directory path of pre-trained model and log files saved.
        'epoch': 30500,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': 10,
    'test_batch_size': 10,
    'augmentation_enable': False,
    'aug_factor': 8,
    'aug_batch_size': 400,
    'test_data_load': {
        'enable': True,
        'filename': './vrp100_test_seed1234.pt'
    },
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'test_cvrp100',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    # create_logger(**logger_params)
    # _print_config()

    tester = Tester(env_params=env_params,
                      model_params=model_params,
                      tester_params=tester_params)

    # copy_all_src(tester.result_folder)

    avg_obj = tester.run()
    
    return avg_obj


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    import sys
    import os
    from gen_inst import dataset_conf

    print("[*] Running ...")

    problem_size = int(sys.argv[1])
    mood = sys.argv[3]
    assert mood in ['train', 'val', "test"]
    
    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.pt")):
        from gen_inst import generate_datasets
        generate_datasets()
    
    if not os.path.isfile(os.path.join(basepath, "checkpoints/checkpoint-30500.pt")):
        raise FileNotFoundError("No checkpoints found. Please see the readme.md and download the checkpoints.")

    if mood == 'train':
        dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
        tester_params['test_data_load']['filename'] = dataset_path
        tester_params['test_episodes'] = 10
        tester_params['test_batch_size'] = 10
        env_params['problem_size'] = problem_size
        avg_obj = main()
        print("[*] Average:")
        print(avg_obj)
    
    else:
        for problem_size in dataset_conf['val']:
            dataset_path = os.path.join(basepath, f"dataset/{mood}{problem_size}_dataset.pt")
            tester_params['test_data_load']['filename'] = dataset_path
            tester_params['test_episodes'] = 64
            tester_params['test_batch_size'] = 64
            env_params['problem_size'] = problem_size
            avg_obj = main()
            print(f"[*] Average for {problem_size}: {avg_obj}")
