import os 
import sys
sys.path.append("../../")


import torch
import numpy as np


from utils import Logger, set_seed, read_config_from_yaml
from model import WaveFunction_Koopman_dynamics, Quad_Value_Net

from train import train_koopman_dynamics, train_value_net
from roll_out import roll_out
from dataset import RLDataset
import controlgym as gym

print("torch version: ", torch.__version__)
set_seed(2024)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = read_config_from_yaml("config.yaml")

# Create environment
env_name = config["env_name"]
process_noise_cov = config["process_noise_cov"]
sensor_noise_cov = config["sensor_noise_cov"]
n_state = config["n_state"]
n_obs = config["n_observation"]
n_action = config["n_action"]
action_limit = config["action_limit"]
obs_limit = config["observation_limit"]
R_weight = config["R_weight"]
Q_weight = config["Q_weight"]
env = gym.make(env_name, process_noise_cov=process_noise_cov, 
                         sensor_noise_cov=sensor_noise_cov,
                         n_state=n_state, n_observation=n_obs, n_action=n_action,
                         action_limit=action_limit, 
                         observation_limit=obs_limit,
                         R_weight=R_weight, 
                         Q_weight=Q_weight)
env_info = {}
state_dim = env.observation_space.shape[0] 
act_dim = env.action_space.shape[0]
    
# Sample Trajectories
num_trajectories = config["num_trajectories"]
traj_length = config["traj_length"]
roll_out_sampler = roll_out(env, env_info)

# Create Koopman Dynamics Model
hidden_dim = config["hidden_dim"]
num_steps = config["num_steps"]
num_mid_layers = config["num_mid_layers"]
pos_dim = int(state_dim / 2)
velocity_dim = int(state_dim / 2)
Kdys_model = WaveFunction_Koopman_dynamics(pos_dim=pos_dim, velocity_dim=velocity_dim, hidden_dim=hidden_dim, act_dim=act_dim, num_steps=num_steps).to(device)

# Save Path (all results)
post_fix = "{}_hidden_{}".format(env_name, hidden_dim)
main_path = "../training_results/{}".format(env_name)
save_path = main_path + "/" + post_fix
if not os.path.exists(save_path):
    os.makedirs(save_path)

print("[INFO] Collecting trajectories")
low = env.observation_space.low
high = env.observation_space.high
if config["sample_trajectories"]:
    train_buffer = roll_out_sampler.roll_out_sampler_with_env(num_trajectories, traj_length, random_action=True)
    np.save(os.path.join(main_path, "train_buffer.npy"), train_buffer)
else:
    train_buffer = np.load(os.path.join(main_path, "train_buffer.npy"))

# Create Dataset
train_dataset = RLDataset(num_steps, state_dim, act_dim, 
                          state_hidden_dim=hidden_dim, act_hidden_dim=8,
                          sample_traj_length=traj_length, 
                          capacity=num_trajectories*(traj_length-num_steps-1))
train_dataset.obtain_data_from_obs_buffer(train_buffer)
print("number of training samples: ", train_dataset.__len__())

# Start Training Dynamics Model
Kdys_model.to(device)
batch_size = config["batch_size"]
num_epochs = config["num_epochs"]
lambda_ = config["lambda"]
beta = config["beta"]
gradclip = config["gradclip"]
lr = config["learning_rate"]
lr_decay_step = config["lr_decay_step"]
lr_decay = config["lr_decay"]


if config["train_dynamics"]:
    # Create Logger 
    metric_name = ["loss_forward", "loss_identity", "loss_distance_preserving"]
    train_logger = Logger(save_path + "/train_log.csv", metric_name)
    test_logger = Logger(save_path + "/test_log.csv", metric_name)
    Kdys_model.load_state_dict(torch.load(os.path.join(save_path, "KoopmanDynamics.pt")))
    train_koopman_dynamics(Kdys_model, train_dataset, num_epochs=num_epochs, 
                        batch_size=batch_size, device=device, 
                        model_save_path=save_path, 
                        lamb=lambda_, beta=beta, gradclip=gradclip,
                        lr=lr, lr_decay_step=lr_decay_step, lr_decay=lr_decay,
                        train_logger=train_logger, test_logger=test_logger)

# Load trained model parameters
Kdys_model.load_state_dict(torch.load(os.path.join(save_path, "KoopmanDynamics.pt")))
Kz = torch.load(os.path.join(save_path, "Kz.pt"))
Jz_B = torch.load(os.path.join(save_path, "Jz_B.pt"))
Kdys_model.koopman.Kz = Kz.to(device)
Kdys_model.koopman.Jz_B = Jz_B.to(device)
Kdys_model = Kdys_model.to(device)

# Train Value Function
value_net = Quad_Value_Net(hidden_dim)
value_net = value_net.to(device)


train_buffer = np.load("train_reward_buffer.npy")
# Create Logger
num_value_epochs = config["num_value_epochs"]
value_net_lr = config["value_net_lr"]
value_net_lr_decay_step = config["value_net_lr_decay_step"]
value_net_lr_decay = config["value_net_lr_decay"]


train_value_net_dataset = RLDataset(num_steps=1, 
                                    state_dim=state_dim, 
                                    act_dim=act_dim, 
                                    state_hidden_dim=hidden_dim, 
                                    act_hidden_dim=8,
                                    sample_traj_length=traj_length, 
                                    capacity=num_trajectories*(traj_length-num_steps-1))
train_value_net_dataset.obtain_data_from_obs_buffer(train_buffer)


# Start Training Value Function
if config["train_value"]:
    train_value_logger = Logger(save_path + "/train_value_log.csv", ["td_error"])
    train_value_net(value_net, Kdys_model, train_value_net_dataset, 
                        number_steps=1, num_epochs=num_value_epochs, 
                        device=device,
                        save_model_path=save_path,
                        lr=5e-4,
                        lr_decay_rate=value_net_lr_decay,
                        lr_decay_step=value_net_lr_decay_step,
                        batch_size=256, 
                        logger=train_value_logger)

print("[INFO] Training Completed")