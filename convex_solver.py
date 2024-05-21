'''
Filename: convex_solver.py
Author: Yiming Yang (zcahyy1@ucl.ac.uk)
Created: 24/01/2024
Description:
    This file contains the class for convex solver for Quadratic problem formed by 
    Jz_B, dv/dz, and LQC term.
'''

import cvxpy as cp
import numpy as np
import torch


def create_convex_solver(env_info, gamma=0.99):
    assert gamma < 1.0, 'discount factor should be less than 1.0'
    print('[INFO] Cretateing the Convex Solver for {}'.format(env_info['Name']))

    action = cp.Variable(env_info["Action Space"].shape[0])
    action_low = torch.tensor(env_info["Action Space"].low, dtype=torch.float32)
    action_high = torch.tensor(env_info["Action Space"].high, dtype=torch.float32)
    conditions = [action >= action_low] \
                + [action <= action_high]
    
    # Coefficeint for Linear Quadratic Cost
    if env_info['Name'] == 'MountainCarContinuous-v0':
        LQC_constant = 0.1
    elif env_info['Name'] == 'Pendulum-v1':
        LQC_constant = 0.01
    elif env_info['Name'] == 'wave':
        LQC_constant = 2.0
    elif env_info['Name'] == 'goal_oriented_control/Lorenz63-v0':
        LQC_constant = 0.01
    elif env_info['Name'] == 'pendulum_raw_image':
        LQC_constant = 0.01
    else:
        raise NotImplementedError('The Coefficeint for Linear Quadratic Cost is not known for {}'.format(env_info['Name']))


    class convex_solver:
        def __init__(self) -> None:
            self.conditions = conditions            
            self.gamma = gamma
            self.action_variable = action
            self.LQC_constant = LQC_constant


        def solve(self, Value_Jacobain_z, J_zB):
             constant = self.gamma*Value_Jacobain_z.T @ J_zB 
             test_h = -self.LQC_constant * np.eye(self.action_variable.shape[0]) 
             if len(constant.shape) == 0:
                problem = cp.Problem(cp.Maximize(
                          constant * self.action_variable + \
                        cp.quad_form(self.action_variable, test_h,
                        )),
                        self.conditions)
             else:
                problem = cp.Problem(cp.Maximize(
                          constant @ self.action_variable + \
                        cp.quad_form(self.action_variable, test_h,
                        )),
                        self.conditions)
             problem.solve()
             opt_action = torch.tensor(self.action_variable.value, dtype=torch.float32)
             return opt_action 

        def solve_in_batch(self, Value_Jacobain_z, J_zB):
            N_batch = J_zB.shape[0]
            opt_action_batch = np.zeros((N_batch, self.action_variable.shape[0]))
            for i in range(N_batch):
                opt_action = self.solve(Value_Jacobain_z[i], J_zB[i])
                opt_action_batch[i] = opt_action
            opt_action_batch = torch.tensor(opt_action_batch, dtype=torch.float32)
            return opt_action_batch
    
    return convex_solver

