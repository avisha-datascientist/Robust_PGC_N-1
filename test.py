import os
import csv
import json
import random
import pathlib
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from grid2op.Parameters import Parameters
from grid2op.Episode import EpisodeData
from custom_reward import *
from agent import Agent
from agent_PDC import Agent_PDC
from train import TrainAgent
from train_PDC import TrainAgent_PDC
from commons import CommonFunctions
import torch.multiprocessing as mp
from train import TrainAgent
import matplotlib.cbook
import wandb
import warnings
import pickle
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


ENV_CASE = {
    '5': 'rte_case5_example',
    'sand': 'l2rpn_case14_sandbox',
    'wcci': 'l2rpn_wcci_2020'
}

DATA_SPLIT = {
    '5': ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    'sand': (list(range(0, 40*26, 40)), list(range(1, 100*10+1, 100)), []),# list(range(2, 100*10+2, 100))),
    'wcci': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
            list(range(2880, 2890)), [])
}

MAX_FFW = {
    '5': 5,
    'sand': 26,
    'wcci': 26
}


def cli():
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--case', type=str, default='5', choices=['sand', 'wcci', '5'])
    parser.add_argument('-gpu', '--gpuid', type=int, default=0)

    parser.add_argument('-ml', '--memlen', type=int, default=50000)
    parser.add_argument('-nf', '--nb_frame', type=int, default=100000,
                        help='the total number of interactions')
    parser.add_argument('-ts', '--test_step', type=int, default=300,
                        help='the interaction number for next evaluation')
    
     
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
    parser.add_argument('-tu', '--target_update', type=int, default=1,
                        help='period of target update')
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='the weight of soft target update')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('-n', '--name', type=str, default='untitled')
    parser.add_argument('-rn', '--run_num', type=int, default=1)
    parser.add_argument('-eps', '--epsilon', type=float, default=0.0)
    parser.add_argument('-mlp', '--multiprocessing_flag', type=int, default=0)
    parser.add_argument('-workers', '--num_workers', type=int, default=1)

    args = parser.parse_args()
    args.actor_lr = args.critic_lr = args.embed_lr = args.alpha_lr = args.lr
    return args

def log_params(args, path):
    f = open(os.path.join(path, "param.txt"), 'w')
    for key, val in args.__dict__.items():
        f.write(key + ': ' + str(val) + "\n")
    f.close()
    with open(os.path.join(path, 'param.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f)

def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[case]):
            with open(os.path.join(path, f'{i}_{j}.json'), 'r', encoding='utf-8') as f:
                a = json.load(f)
                res[(i,j)] = (a['dn_played'], a['donothing_reward'], a['donothing_nodisc_reward'])
            if i >= 2880: break
    return res

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        print('gpu is not available')


def set_new_parameters():
    env_new_params = Parameters()
    env_new_params.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    env_new_params.NB_TIMESTEP_RECONNECTION = 12
    env_new_params.NB_TIMESTEP_COOLDOWN_LINE = 3
    env_new_params.NB_TIMESTEP_COOLDOWN_SUB = 3
    env_new_params.HARD_OVERFLOW_THRESHOLD = 200.0

    return env_new_params

def create_environment():
    env_ex = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(), other_rewards={'loss': LossReward})
    env_ex.deactivate_forecast()
    
    return env_ex

def create_agent(env, test_env, cversion_path, args):
    if args.multiprocessing_flag:
        my_agent = Agent_PDC(env,test_env, **vars(args))
    else:
        my_agent = Agent(env, cversion_path, **vars(args))
    mean = torch.load(os.path.join(env_path, 'mean.pt'))
    std = torch.load(os.path.join(env_path, 'std.pt'))
    my_agent.load_mean_std(mean, std)

    return my_agent

def get_pickling_errors(obj,seen=None):
    print(obj)
    if seen == None:
        seen = []
    try:
        state = obj.__getstate__()
        print(state)
    except AttributeError:
        print("Attributeerror")
        return
    if state == None:
        print("State is none")
        return
    if isinstance(state,tuple):
        if not isinstance(state[0],dict):
            state=state[1]
        else:
            state=state[0].update(state[1])
    result = {}    
    for i in state:
        try:
            pickle.dumps(state[i],protocol=2)
        except pickle.PicklingError:
            if not state[i] in seen:
                seen.append(state[i])
                result[i]=get_pickling_errors(state[i],seen)
    return result

def run_actor(trainer, **kwargs):
    trainer.run_actor_workers(**kwargs)

def run_learner(agent, **kwargs):
    agent.run_agent_process(**kwargs)

def run_parallel(trainer, agent, num_actors, seed, nb_frame, test_step, train_chronics, valid_chronics, output_dir, model_name, max_ffw, dn_ffw, ep_infos):
    mp.set_start_method('spawn', force=True)  
    mp.freeze_support()

    # any other data to share between process?
    shared_kwargs = {
        'shared_memory': mp.Queue(),
        'chronic_priority': mp.Manager().list(range(1500)),
        'shared_weights': mp.Manager().dict(),
        'learnings_num': mp.Array(typecode_or_type='I', size_or_initializer=1)
    }
        
    learner_kwargs = dict(
        agent=agent,  
        valid_chronics=valid_chronics,
        nb_frame=nb_frame, 
        test_step=test_step,
        max_ffw=max_ffw,
        dn_ffw=dn_ffw,
        ep_infos=ep_infos,
        output_dir=output_dir,
        model_name=model_name,
        **shared_kwargs,
    )
    
    r1 = get_pickling_errors(agent)
    r2 = get_pickling_errors(valid_chronics)
    r3 = get_pickling_errors(nb_frame)
    r4 = get_pickling_errors(test_step)
    r5 = get_pickling_errors(max_ffw)
    r6 = get_pickling_errors(dn_ffw)
    r7 = get_pickling_errors(ep_infos)
    r8 = get_pickling_errors(output_dir)
    r9 = get_pickling_errors(model_name) 
    
    processes = [mp.Process(target=run_learner, kwargs=learner_kwargs)]
        
   
    for actor_id in range(num_actors):
        actor_kwargs = dict(
            trainer=trainer,
            actor_id=actor_id,
            seed=seed, 
            train_chronics=train_chronics,
            max_ffw=max_ffw,
            output_dir=output_dir,
            model_name=model_name,
            **shared_kwargs,
        )
        processes.append(mp.Process(target=run_actor, kwargs=actor_kwargs))

    for pi in range(len(processes)):
        print("process started:", pi)
        processes[pi].start()

    for p in processes:
        p.join()
        


if __name__ == '__main__':
    args = cli()
    seed_everything(args.seed)

    # settings
    model_name = f'{args.name}_{args.seed}'
    print('model name: ', model_name)
    print('Printing cuda version')
    print(torch.version.cuda) 
    base_path = '.'
    OUTPUT_DIR = 'result'
    DATA_DIR = 'data'
    # change serialization directory accordingly
    SERIALIZATION_DIR = 'serialized/agents'
 
    result_dir = os.path.join(base_path, OUTPUT_DIR)
    output_result_dir = os.path.join(result_dir, model_name)
    model_path = os.path.join(output_result_dir, 'model')
    cversion_path = os.path.join(model_path, 'current_version')
    serialized_agent_dir = os.path.join(base_path, SERIALIZATION_DIR) 
    tensorboard_logs = os.path.join(output_result_dir, 'tensorboard_summaries')
    #serialized_agent_dir = os.path.join(serialize_dir, 'SMAAC')
         
    if torch.cuda.is_available():	
        print('GPU is being used')
        torch.cuda.set_device(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_name = ENV_CASE[args.case]
    data_dir_path = os.path.join(base_path, DATA_DIR)
    env_path = os.path.join(data_dir_path, env_name)
    chronics_path = os.path.join(env_path, 'chronics')
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.case]
    dn_json_path = os.path.join(env_path, 'json')
    
    # select chronics
    dn_ffw = read_ffw_json(dn_json_path, train_chronics + valid_chronics, args.case)

    ep_infos = {}
    
    if os.path.exists(dn_json_path):
        for i in list(set(train_chronics+valid_chronics)):
            with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
                ep_infos[i] = json.load(f)
    if not os.path.exists(os.path.abspath(output_result_dir.strip())):
        os.makedirs(output_result_dir, exist_ok = True)
        os.chown(output_result_dir, os.stat(output_result_dir).st_uid, os.stat(output_result_dir).st_gid)
        os.makedirs(model_path, exist_ok = True)
        log_params(args, output_result_dir)
        os.makedirs(cversion_path,exist_ok = True)
        # Uncomment when using w/o parallelization
        os.makedirs(tensorboard_logs, exist_ok = True)

    env = create_environment()
    test_env = create_environment()
    env.seed(args.seed)
    test_env.seed(59)
    chronic_num = len(test_chronics)
            
    # to update the env parameters
    env.change_parameters(set_new_parameters())
    test_env.change_parameters(set_new_parameters())

    print(env.parameters.__dict__)  
    my_agent = create_agent(env, test_env, cversion_path, args)
    
    if args.multiprocessing_flag:
        trainer = TrainAgent_PDC(my_agent, env, test_env, device, dn_json_path, serialized_agent_dir, dn_ffw, ep_infos, None)
        run_parallel(trainer, my_agent, args.num_workers, args.seed, args.nb_frame, args.test_step, train_chronics, valid_chronics, OUTPUT_DIR, model_name, MAX_FFW[args.case], dn_ffw, ep_infos)
    else:
        
        # specify agent
        trainer = TrainAgent(my_agent, env, test_env, device, dn_json_path, serialized_agent_dir, dn_ffw, ep_infos, None, args.epsilon)
            
        trainer.train(
            args.seed, args.nb_frame, args.test_step,
            train_chronics, valid_chronics, output_result_dir, model_path, MAX_FFW[args.case])
        trainer.agent.save_model(model_path, 'last')
        trainer.agent.save_most_recent()
        
