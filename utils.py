'''
Filename: utils.py
Created: 24/09/2023
Description:
    This file contains some utility functions.
'''

import os 
import numpy as np
import torch
import gymnasium as gym
from wrappers import modified_pendulum
import yaml

'''
Utility functions for gym environments
'''
def query_environment_info(env, env_name, render_mode=None):
    print('[INFO] Creating the environment: {}'.format(env))
    name = env_name
    env_info = {"Name" : name,
                "Action Space" : env.action_space,
                "Observation Space" : env.observation_space,
                "Reward Range"      : env.reward_range,
                "Default Reset Noise Scale" : 0.1}

    print(f"Name: {name}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Reward Range: {env.reward_range}")
    print('')
    return env_info
    
def query_environment(name, render_mode=None):
    print('[INFO] Creating the environment: {}'.format(name))
    # env = gym.make(name, render_mode=render_mode, angle_representation='euler')
    env = gym.make(name, render_mode=render_mode)
    # env = gym.make(name, render_mode)
    spec = gym.spec(name)
    env_info = {"Name" : name,
                "Action Space" : env.action_space,
                "Observation Space" : env.observation_space,
                "Max Episode Steps" : spec.max_episode_steps,
                "Nondeterministic"  : spec.nondeterministic,
                "Reward Range"      : env.reward_range,
                "Reward Threshold"  : spec.reward_threshold,
                "Default Reset Noise Scale" : 0.1}

    print(f"Name: {name}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Max Episode Steps: {spec.max_episode_steps}")
    print(f"Nondeterministic: {spec.nondeterministic}")
    print(f"Reward Range: {env.reward_range}")
    print(f"Reward Threshold: {spec.reward_threshold}")
    print('')

    if name == 'Pendulum-v1':  
        env = modified_pendulum(env) 

    return env, env_info

         

'''
Utility functions for Data Collector
'''
def padding(pre_pad, traj_length, num_traj, dim, expand_dim=False):
    padded_seq = np.zeros((num_traj, traj_length, dim))
    for i in range(num_traj):
        diff = traj_length - len(pre_pad[i])
        pad = np.array(pre_pad[i]) if not expand_dim else np.array(pre_pad[i])[:,None]
        padded_seq[i] = np.pad(pad, ((0,diff),(0,0)), 'constant', constant_values=0.0)
    return padded_seq 


def transform_into_theta(states):
   if len(states.shape) == 3:
      # states: [batch_size, time_steps, state_dim]
      transformed_states = np.zeros((states.shape[0], states.shape[1], states.shape[2]-1))

      theta = np.arctan2(states[:,:,1],states[:,:,0])
      theta_dot = states[:,:,2]

      transformed_states[:,:,0] = theta
      transformed_states[:,:,1] = theta_dot
      return transformed_states
   else:
      return np.array([np.arctan2(states[1],states[0]), states[2]])
   
'''
Logger Functions
'''

class Logger:
    def __init__(self, log_file=None, 
                 metrics_names=['loss_forward', 'loss_identity', 'loss_distance_preserving']):
        self.log_file = log_file
        self.metric_names = metrics_names
        
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            with open(log_file, 'w') as f:
                f.write(','.join(['epoch'] + self.metric_names) + '\n')
        else:
            raise NotImplementedError("[WARNING] log_file path is not provided")
        
        print('[INFO] Logging to {}'.format(log_file))
    
    def log_metrics(self, epoch, **metrics):
        log_entries = [f"{epoch}"]
        for name in self.metric_names:
            value = metrics.get(name, "N/A")
            log_entries.append(str(value))
            print('[INFO] Epoch: {}, {}: {}'.format(epoch, name, value))

        with open(self.log_file, 'a') as f:
            f.write(','.join(log_entries) + '\n')
        print(' ')
            

'''
Other Utility Functions
'''
def set_seed(seed=2023):
    print('[INFO] Setting the seed to {}'.format(seed))
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_device():
    if torch.cuda.device_count()>0:
        device = torch.device('cuda:0')
        print("[INFO] Connected to a GPU")
    else:
        print("[INFO] Using the CPU")
        device = torch.device('cpu')
    return device

def is_nd(mat):
    # check if a matrix is negative definite
    return (torch.linalg.eigvals(mat).real<0).all()

    
    
def read_config_from_yaml(config_path):
    with open(config_path,'r') as f:
        data = yaml.safe_load_all(f)
        loaded_data = list(data)[0]
    print('[INFO] Loaded data from {}'.format(config_path))
    return loaded_data

