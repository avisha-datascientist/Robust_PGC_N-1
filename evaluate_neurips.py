import os
import csv
import json
import random
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from custom_reward import *
from agent import Agent
from train import TrainAgent
from matplotlib import pyplot as plt
import matplotlib.cbook
import wandb
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


ENV_CASE = {
    '5': 'rte_case5_example',
    'sand': 'l2rpn_case14_sandbox',
    'wcci': 'l2rpn_wcci_2020',
    'neurips': 'l2rpn_neurips_2020_track1_small'
}

DATA_SPLIT = {
    '5': ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    'sand': (list(range(0, 40*26, 40)), list(range(1, 100*10+1, 100)), list(range(2, 100*10+2, 100))),
    'wcci': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
            list(range(2880, 2890)), [18, 241, 495, 738, 991, 1453, 1718, 1943, 2205, 2404]),
    'neurips':(list(range(0,30)))
}

MAX_FFW = {
    '5': 5,
    'sand': 26,
    'wcci': 26
}


def cli():
    parser = ArgumentParser()
    parser.add_argument('-c', '--case', type=str, default='wcci', choices=['sand', 'wcci', '5', 'neurips'])
    parser.add_argument('-gpu', '--gpuid', type=int, default=0)

    parser.add_argument('-hn', '--head_number', type=int, default=8,
                        help='the number of head for attention')
    parser.add_argument('-sd', '--state_dim', type=int, default=128,
                        help='dimension of hidden state for GNN')
    parser.add_argument('-nh', '--n_history', type=int, default=6,
                        help='length of frame stack')
    parser.add_argument('-do', '--dropout', type=float, default=0.)
    
    parser.add_argument('-r', '--rule', type=str, default='c', choices=['c', 'd', 'o', 'f'],
                        help='low-level rule (capa, desc, opti, fixed)')
    parser.add_argument('-thr', '--threshold', type=float, default=0.1,
                        help='[-1, thr) -> bus 1 / [thr, 1] -> bus 2')
    parser.add_argument('-dg', '--danger', type=float, default=0.9,
                        help='the powerline with rho over danger is regarded as hazardous')
    parser.add_argument('-m', '--mask', type=int, default=5,
                        help='this agent manages the substations containing topology elements over "mask"')
    parser.add_argument('-mhi', '--mask_hi', type=int, default=19)
    parser.add_argument('-mll', '--max_low_len', type=int, default=19)
    
    parser.add_argument('-l', '--last', action='store_true')
    parser.add_argument('-n', '--name', type=str, required=True)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = cli()    
    # settings
    model_name = args.name
    print('model_name: ', model_name)

    base_path = '.'
    OUTPUT_DIR = 'result'
    DATA_DIR = 'data'
    result_dir = os.path.join(base_path, OUTPUT_DIR)   
    output_result_dir = os.path.join(result_dir, model_name)
    model_path = os.path.join(output_result_dir, 'model')

    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluation_kit = "starting_k_test"
    env_name = ENV_CASE[args.case]
    data_r_dir = os.path.join(base_path, DATA_DIR)
    env_path = os.path.join(data_r_dir, evaluation_kit, 'L2RPN_neurips2020_track1_starting_kit', 'input_data_local')
    chronics_path = os.path.join(env_path, 'chronics')
    cversion_path = os.path.join(model_path, 'current_version')
    
    test_chronics = DATA_SPLIT[args.case]
    mean_std_path = os.path.join(data_r_dir, env_name)
    #train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.case]
    
    mode = 'last' if args.last else 'best'

    env = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
                other_rewards={'loss': LossReward})
    env.deactivate_forecast()
    env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    env.parameters.NB_TIMESTEP_RECONNECTION = 12
    env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    print(env.parameters.__dict__)

    my_agent = Agent(env, cversion_path, **vars(args))
    mean = torch.load(os.path.join(mean_std_path, 'mean.pt'))
    std = torch.load(os.path.join(mean_std_path, 'std.pt'))
    my_agent.load_mean_std(mean, std)
    my_agent.load_model(model_path, mode)

    trainer = TrainAgent(my_agent, env, env, device, None, None, None, None, None)
    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
    plot_topo = True
    trainer.evaluate_neurips(model_name, test_chronics, output_result_dir, mode, plot_topo)
