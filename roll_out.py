'''
Filename: roll_out.py
Author: Yiming Yang (zcahyy1@ucl.ac.uk)
Created: 24/01/2024
Description:
    This file contains the class for roll out Trajectory from given Environment.
'''

import torch 
import numpy as np
import tqdm
from utils import padding

class roll_out:
    def __init__(self, env, env_info) -> None:
          self.name = "{}_roll_out".format(env_info['Name'])

          self.env = env
          self.action_dim = env_info['Action Space'].shape[0]
          self.state_dim = env_info['Observation Space'].shape[0] if env_info['Name'] != 'Pendulum-v1' else env_info['Observation Space'].shape[0] - 1
          self.default_reset_noise_scale = env_info['Default Reset Noise Scale']


    def roll_out_sampler_with_env(self, num_trajactories=10, max_time_step=100, policy=None,
                                  koopman_dynamics_model=None, value_net=None, reward_net=None,
                                  random_action=True, start_noise_scale=5, action_noise=None, device=torch.device('cpu')):
        states = [[] for _ in range(num_trajactories)]
        actions = [[] for _ in range(num_trajactories)]
        rewards = [[] for _ in range(num_trajactories)]
        masks = [[] for _ in range(num_trajactories)]

        self.env.reset_noise_scale = start_noise_scale
        if random_action:
            def select_action(obs):
                return self.env.action_space.sample()
        else:
            assert policy is not None, "Policy is not provided"
            assert koopman_dynamics_model is not None, "Koopman Dynamics Model is not provided"
            assert value_net is not None, "Value Net is not provided"
            assert reward_net is not None, "Reward Net is not provided"
            koopman_dynamics_model.eval()
            value_net.eval()
            reward_net.eval()
            def select_action(obs):
                # obs = self.data_transform(obs)
                obs = torch.tensor(obs, dtype=torch.float32).to(device)
                action = policy.select_actions(koopman_dynamics_model, value_net, reward_net,
                                            obs, if_batch=False, boostrap=False)
                if action_noise is not None:
                    action = action + np.random.normal(0, action_noise, size=action.shape)
                return np.clip(action.detach().numpy(), -1, 1)
        
        for i in tqdm.tqdm(range(num_trajactories)):
            t = 0
            obs = self.env.reset()
            if len(obs) == 1:
                obs = obs
            else:
                obs = obs[0]
                
            while t < max_time_step:
                t += 1
                action = select_action(obs)
                next_obs, reward, done, _, _ = self.env.step(action)

                states[i].append(obs)
                actions[i].append(action)
                rewards[i].append(reward)
                masks[i].append(1)
                obs = next_obs

                if done:
                    break

        self.env.reset_noise_scale = self.default_reset_noise_scale
        pad_states = padding(states, max_time_step, num_trajactories, self.state_dim)
        pad_actions = padding(actions, max_time_step, num_trajactories, self.action_dim)
        pad_rewards = padding(rewards, max_time_step, num_trajactories, 1, expand_dim=True)
        pad_masks = padding(masks, max_time_step, num_trajactories, 1, expand_dim=True)

        data = np.concatenate([pad_states, pad_actions, pad_rewards, pad_masks], axis=-1)

        return data
    

