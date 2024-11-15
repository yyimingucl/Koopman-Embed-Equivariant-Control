import os
import torch
import numpy as np
import cvxpy as cp
from cvxpy import quad_form
import tqdm

def MPC_QP(cur_z, goal_z, Kz, J_zB, Q, R, action_limit, h):
    n_z = cur_z.shape[0]
    n_a = J_zB.shape[1]

    z = cp.Variable( (h+1, n_z) )
    a = cp.Variable( (h, n_a) )

    cost = 0
    constraints = [z[0]==cur_z]

    for t in range(1, h+1):
        constraints.append(a[t-1]<=action_limit)
        constraints.append(a[t-1]>=-action_limit)
        constraints.append(z[t] == Kz @ z[t-1] + J_zB @ a[t-1])
        cost += quad_form(a[t-1], R)
    
    cost += quad_form(z[h-1]-goal_z, Q)


    objective = cp.Minimize(cost)
    prob = cp.Problem(objective, constraints)
    result = prob.solve()

    return a[0].value

    


class Koopman_MPC_QP:
    def __init__(self, env, Koopman_Dynamics, goal=None):
        self.env = env
        self.action_limit = env.action_space.high[0]

        Q_weight = self.env.Q[0,0] 
        R_weight = self.env.R[0,0]  

        self.Q = np.identity(Koopman_Dynamics.hidden_dim) * Q_weight
        self.R = np.identity(Koopman_Dynamics.act_dim) * R_weight
        
        self.Kz, self.Jz_B = Koopman_Dynamics.koopman.Kz.detach().numpy(), Koopman_Dynamics.koopman.Jz_B.detach().numpy()
        self.to_z = Koopman_Dynamics.encoder

        if goal is not None:
            with torch.no_grad():
                self.goal = self.to_z(goal).detach().numpy()
        else:
            self.goal = np.zeros(Koopman_Dynamics.hidden_dim)

    

    
    def perform_MPC(self, H, init_state=None):
        # subsample_idx = np.arange(0, self.env.n_state, 2)
        reward = 0
        state_traj = np.zeros((self.env.n_state, self.env.n_steps + 1))
        state_traj[:, 0] = init_state
        action_traj = np.zeros((self.env.n_action, self.env.n_steps))
        total_reward = []
        # H: Control Horizon
        with torch.no_grad():
            if init_state is not None:
                self.env.reset(init_state)
                init_state = torch.from_numpy(init_state[None,:]).float()
                cur_z = self.to_z(init_state).squeeze(0).detach().numpy() 
            else:
                s = self.env.reset()
                s = s[0] if type(s) is tuple else s
                # s = s[subsample_idx]
                s = torch.from_numpy(s[None,:]).float() 
                cur_z = self.to_z(s).squeeze(0).detach().numpy()
            
            for h in tqdm.tqdm(range(H), desc="Performing Koopman MPC"):
                action = MPC_QP(cur_z, self.goal, self.Kz, self.Jz_B, self.Q, self.R, self.action_limit, 3)
                next_state, reward, done, _, _ = self.env.step(action)

                next_state_t = torch.from_numpy(next_state[None,:]).float()
                cur_z = self.to_z(next_state_t).squeeze(0).detach().numpy()
                if done:
                    break

                state_traj[:, h+1] = next_state
                action_traj[:, h] = action
                total_reward.append(reward)
        return total_reward, state_traj, action_traj

            

        

        