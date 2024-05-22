'''
Filename: dataset.py
Created: 24/01/2024
Description:
    This file contains the class for dataset for training.
'''


from torch.utils.data import Dataset
import torch
import numpy as np

class RLDataset(Dataset):
    def __init__(self, num_steps:int, state_dim:int, act_dim:int, state_hidden_dim:int, act_hidden_dim:int, 
                 sample_traj_length:int=100, capacity:int=10000) -> None:
        # num_steps: length of training sequence *** but sample num_steps + 1 ***
        # sample_traj_length: length of sampled trajectory

        self.num_steps = num_steps

        self.state_dim = state_dim
        self.state_hidden_dim = state_hidden_dim

        self.act_dim = act_dim
        self.act_hidden_dim = act_hidden_dim
        
        self.num_slot_per_traj = sample_traj_length - num_steps 

        self.sample_traj_length = sample_traj_length
        self.capacity = capacity

        self.pool = []
        self.position = 0

    def __len__(self):
        return len(self.pool)
    
    def __getitem__(self, idx):
        # pool_id = np.random.choice(len(self.pool), replace=False)

        data = np.array(self.pool[idx])
        data = torch.tensor(data, dtype=torch.float32)

        # Dim (state_dim+act_dim+reward_dim) x Time
        inital_states = data[:-1, :self.state_dim]
        actions = data[:-1, self.state_dim:self.state_dim+self.act_dim]
        following_states = data[1:, :self.state_dim]
        reward = data[:-1,-2]

        masks = data[:-1,-1]
        if 0 in masks:
            zero_id = np.where(masks==0)[0][0]
            masks[zero_id:] = 0

        return inital_states, actions, following_states, reward, masks

    def obtain_data_from_obs_buffer(self, obs_buffer:np.array):
        N = obs_buffer.shape[0]
        assert obs_buffer.shape == (N, self.sample_traj_length, self.state_dim + self.act_dim + 1 + 1), \
            'obs_buffer {} does not match ({},{},{})'.format(obs_buffer.shape, N, self.sample_traj_length, self.state_dim + self.act_dim + 1 + 1)
        for i in range(N):
            for j in range(self.num_slot_per_traj):
                self.push_into_pool(obs_buffer[i, j:j+self.num_steps+1, :])
        
    def push_into_pool(self, sequence_data):
        # sequence_data: [state_dim + act_dim + 1, sample_traj_length]
        assert sequence_data.shape == (self.num_steps+1, self.state_dim + self.act_dim + 1 + 1), \
            'sequence_data {} does not match ({},{})'.format(sequence_data.shape, self.num_steps+1, self.state_dim + self.act_dim + 1 + 1)
        if len(self.pool) < self.capacity:
            self.pool.append(None)
        self.pool[self.position] = sequence_data
        self.position = (self.position + 1) % self.capacity

    def clear(self):
        self.memory.clear()
        self.position = 0



if __name__ == '__main__':
    import time
    from torch.nn  import functional as F 

    num_steps = 10
    state_dim = 3
    act_dim = 1
    state_hidden_dim = 10
    act_hidden_dim = 10
    sample_traj_length = 100

    s = time.time()

    ds = RLDataset(num_steps=num_steps, state_dim=state_dim, act_dim=act_dim, 
                        state_hidden_dim=state_dim, act_hidden_dim=act_dim,
                        sample_traj_length=sample_traj_length, capacity=10000)

    obs_buffer = np.random.rand(10, sample_traj_length, state_dim+act_dim+1+1)
    ds.obtain_data_from_obs_buffer(obs_buffer)
    e = time.time()
    print('[INFO] time for loading data: {}s'.format(e-s))

    initial_states, actions, following_states, rewards, masks = next(iter(ds))
    assert initial_states.shape == (num_steps, state_dim), 'initial_states.shape {} does not match with {}'.format(initial_states.shape, (num_steps, state_dim))
    assert actions.shape == (num_steps, act_dim), 'actions.shape {} does not match with {}'.format(actions.shape, (num_steps, act_dim))
    assert following_states.shape == (num_steps, state_dim), 'following_states.shape {} does not match with {}'.format(following_states.shape, (num_steps, state_dim))
    assert rewards.shape == (num_steps,), 'rewards.shape {} does not match with {}'.format(rewards.shape, (num_steps,))
    assert masks.shape == (num_steps,), 'masks.shape {} does not match with {}'.format(masks.shape, (num_steps,))
    assert np.all(initial_states[:,0]>0 and initial_states[:,1]<1), 'initial_states {} does not match with {}'.format(initial_states, '0<normaliza(theta)<1')
    print('[INFO] dimension check passed!')

    error = F.mse_loss(initial_states[1:, :], following_states[:-1, :])
    assert error < 1e-20, 'overlapping state error is {}. state and next_state does not match'.format(error)
    print('[INFO] Error: {}'.format(error))
    print('[INFO] overlapping state check passed!')