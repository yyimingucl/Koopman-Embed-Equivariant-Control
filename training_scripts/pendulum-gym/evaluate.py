import torch
import numpy as np
import tqdm 
import os 
import inspect
import sys
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
parentdir = os.path.dirname(parentdir)
sys.path.insert(0, parentdir) 

from utils import set_seed, query_environment, read_config_from_yaml
from model import Quad_Value_Net, Policy,Koopman_dynamics
from solver import create_solver
from roll_out import roll_out
from dataset import RLDataset
import gymnasium as gym
import matplotlib.pyplot as plt


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
set_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = read_config_from_yaml("config.yaml")


env_name = config["env_name"]
env, env_info = query_environment(env_name) 
state_dim = env.observation_space.shape[0] - 1
act_dim = env.action_space.shape[0]


# save model path
hidden_dim = config["hidden_dim"]
post_fix = "{}_hidden_{}".format(env_name, hidden_dim)
main_path = "../training_results/{}".format(env_name)
save_path = main_path + "/" + post_fix


# Construct and Load Model
num_steps = config["num_steps"]
Kdys_model = Koopman_dynamics(state_dim=state_dim, hidden_dim=hidden_dim, act_dim=act_dim, num_steps=num_steps).to(device)
Kdys_model.load_state_dict(torch.load(save_path + '/KoopmanDynamics.pt', map_location=torch.device('cpu')))
Kz = torch.load(save_path + '/Kz.pt', map_location=torch.device('cpu'))
Jz_B = torch.load(save_path + '/Jz_B.pt', map_location=torch.device('cpu'))
Kdys_model.koopman.Kz = Kz.to(device)
Kdys_model.koopman.Jz_B = Jz_B.to(device)
Kdys_model.to(device)


# Evaluate the predicition of the Koopman Dynamics
roll_out_sampler = roll_out(env, env_info)
num_trajactories=10
max_time_step=20
for i in range(1):
    test_buffer = roll_out_sampler.roll_out_sampler_with_env(num_trajactories=num_trajactories,
                                                             max_time_step=max_time_step, random_action=True)
    test_dataset = RLDataset(num_steps, state_dim, act_dim, 
                             state_hidden_dim=hidden_dim, 
                             act_hidden_dim=8, 
                             sample_traj_length=max_time_step, 
                             capacity=num_trajactories*(max_time_step-num_steps-1))
    test_dataset.obtain_data_from_obs_buffer(test_buffer)

j = np.random.randint(0, num_trajactories)
initial_states, actions, following_states, rewards, dones = test_dataset.__getitem__(j)

pred_next_states = []
for i in range(num_steps):
    state = initial_states[i].to(device).unsqueeze(0)
    action = actions[i].to(device).unsqueeze(0)
    
    next_state = Kdys_model(state, action).squeeze(0).detach().cpu().numpy()
    pred_next_states.append(next_state)

pred_next_states = np.array(pred_next_states)

plt.scatter(initial_states[0,0], initial_states[0,1], c='r', label='initial states', marker="+")
plt.scatter(following_states[:20,0], following_states[:20,1], c='b', label='following states')
plt.scatter(pred_next_states[:20,0], pred_next_states[:20,1], c='g', label='predicted states')
plt.legend()
plt.show()



opt_state = np.array([[0.0 for _ in range(state_dim)]])
opt_state = torch.tensor(opt_state, dtype=torch.float32).to(device)
print(opt_state.shape)
encoded_opt_state = Kdys_model.encoder(opt_state).clone().detach()

value_net = Quad_Value_Net(hidden_dim, encoded_opt_state.clone().detach()).to(device)
value_net.opt_encoded_state = torch.nn.Parameter(encoded_opt_state, requires_grad=False)
value_net.load_state_dict(torch.load(save_path + '/Quad_Value_Net.pt', map_location=torch.device('cpu')))

# Construct Policy
cvx_solver = create_convex_solver(env_info)()
policy = Policy(cvx_solver)


# Start Evaluation
eval_T = config["eval_T"]
eval_episodes = config["eval_episodes"]
eval_rewards = [[] for i in range(eval_episodes)]
eval_traj = np.zeros((eval_episodes, eval_T+1, state_dim))
eval_traj_obs = np.zeros((eval_episodes, eval_T+1, state_dim))
eval_actions = np.zeros((eval_episodes, eval_T+1, act_dim))


for n in tqdm.tqdm(range(eval_episodes)):
    t = 0
    state = env.reset()[0]
    eval_traj_obs[n, t, :] = state
    eval_traj[n, t, :] = env.state
    while t < eval_T:
        t += 1

        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        action = policy.select_actions(Kdys_model, value_net, state, if_batch=False, boostrap=False).detach().cpu().numpy()

        eval_traj_obs[n, t, :] = state.detach().cpu().numpy()
        eval_actions[n, t, :] = action
        next_state, reward, done, _, _ = env.step(action)
        eval_rewards[n].append(reward)
        state = next_state

        eval_traj[n, t, :] = env.state
        
        if done:
            break

eval_rewards = np.array(eval_rewards)
print("Average Reward: ", np.mean(eval_rewards))
np.save(save_path + "/" + "eval_traj.npy", eval_traj)
np.save(save_path + "/" + "eval_actions.npy", eval_actions)
np.save(save_path + "/" + "eval_rewards.npy", eval_rewards)