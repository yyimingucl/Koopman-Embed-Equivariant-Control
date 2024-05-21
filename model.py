'''
Filename: model.py
Author: Yiming Yang (zcahyy1@ucl.ac.uk)
Created: 24/01/2024
Description:
    This file defines deep learning models for the Koopman operator 
    framework for learning dynamical system. 
'''


import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable, functional
 
initial_value_function_regularize = 1e-3
delta_t = 1

'''
Auto Encoder
'''
class Encoder(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, *args, **kwargs) -> None:
        super(Encoder, self).__init__(*args, **kwargs)
        half_dim = int(hidden_dim/2)

        self.encoder =  nn.Sequential(
                                    nn.Linear(state_dim, half_dim),
                                    
                                    nn.Tanh(),
                                    nn.Linear(half_dim, half_dim),
                                    
                                    
                                    nn.Tanh(),
                                    nn.Linear(half_dim, hidden_dim),
                                    )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    
    def forward(self, x:Tensor):
        return self.encoder(x)

    def compute_Jacobian(self, x:Tensor):
        J = functional.jacobian(self.encoder, x)
        return J
    
    def freeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = False
            
    
    def defreeze_encoder(self):
        for param in self.parameters():
            param.requires_grad = True
    


class Decoder(nn.Module):
    def __init__(self, state_dim:int, hidden_dim:int, *args, **kwargs) -> None:
        super(Decoder, self).__init__(*args, **kwargs)
        half_dim = int(hidden_dim/2)
        self.decoder =  nn.Sequential(
                            nn.Linear(hidden_dim, half_dim),

                            nn.Tanh(),
                            nn.Linear(half_dim, half_dim),
                            
                            nn.Tanh(),
                            nn.Linear(half_dim, state_dim),
                            )
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    
    def forward(self, x:Tensor):
        return self.decoder(x)
    
    def freeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def defreeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = True


class WaveFunctionEncoder(nn.Module):
    def __init__(self, pos_dim:int=25, velocity_dim:int=25, hidden_dim:int=16, *args, **kwargs) -> None:
        super(WaveFunctionEncoder, self).__init__(*args, **kwargs)
        self.pos_dim = pos_dim
        self.velocity_dim = velocity_dim 
        self.hidden_dim = hidden_dim
        self.position_enocder = nn.Sequential(
                                    nn.Linear(pos_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 32),
                                    nn.Tanh(),
                                    nn.Linear(32, int(hidden_dim/2)), 
                                    nn.Tanh(),
                                    )
         
        self.velocity_encoder = nn.Sequential(
                                    nn.Linear(velocity_dim, 64),
                                    nn.Tanh(),
                                    nn.Linear(64, 32),
                                    nn.Tanh(),
                                    nn.Linear(32, int(hidden_dim/2)), 
                                    nn.Tanh(),
                                    )
        
        self.joint_encoder = nn.Linear(hidden_dim, hidden_dim)
    
             
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    
    def forward(self, x:Tensor):
        x_position = x[:, :self.pos_dim]
        x_velocity = x[:, self.pos_dim:]
        
        z_position = self.position_enocder(x_position)
        z_velocity = self.velocity_encoder(x_velocity)
        
        z = torch.cat((z_position, z_velocity), dim=-1)
        z = self.joint_encoder(z)
        return z
        
    
    def freeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def defreeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = True


class WaveFunctionDecoder(nn.Module):
    def __init__(self, pos_dim:int=25, velocity_dim:int=25, hidden_dim:int=16, *args, **kwargs) -> None:
        super(WaveFunctionDecoder, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.joint_decoder = nn.Linear(hidden_dim, hidden_dim)  
                                        
        self.position_decoder = nn.Sequential(
                                        nn.Tanh(),
                                        nn.Linear(int(hidden_dim/2), 32),
                                        nn.Tanh(),
                                        nn.Linear(32, 64),
                                        nn.Tanh(),
                                        nn.Linear(64, pos_dim)
                                        )
        self.velocity_decoder = nn.Sequential(
                                        nn.Tanh(),
                                        nn.Linear(int(hidden_dim/2), 32),
                                        nn.Tanh(),
                                        nn.Linear(32, 64),
                                        nn.Tanh(),
                                        nn.Linear(64, velocity_dim)
                                        )
             
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                nn.init.zeros_(m.bias.data)
    
    
    def forward(self, z:Tensor):
        z_joint = self.joint_decoder(z)
        
        z_position = z_joint[:, :int(self.hidden_dim/2)]
        z_velocity = z_joint[:, int(self.hidden_dim/2):]
        
        x_position = self.position_decoder(z_position)
        x_velocity = self.velocity_decoder(z_velocity)
        x = torch.cat((x_position, x_velocity), dim=-1)
        return x
        
        
    
    def freeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = False
    
    def defreeze_decoder(self):
        for param in self.parameters():
            param.requires_grad = True
    

'''
Koopman Operator
'''
# Operation in Latent Space
class KoopmanOp(nn.Module):
    # Remark: delta t = 1 in simple case. 
    # If not simple case, delta t need to be taken into account

    def __init__(self, hidden_dim:int, act_dim:int, num_steps:int, *args, **kwargs) -> None:
        super(KoopmanOp, self).__init__(*args, **kwargs)

        self.num_steps = num_steps
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        self.Kz = None
        self.Jz_B = None

    def compute_Jz_B(self):
        return self.Jz_B.unsqueeze(0).detach()

    def forward(self, z:Tensor, a:Tensor):
        # z: [hidden_dim]
        # a: [act_dim]

        z_1 = torch.mm(z, self.Kz.T)
        z_2 = torch.mm(a, self.Jz_B.T)

        z_next = z + (z_1 + z_2)*delta_t
        return z_next

    def batch_forward(self, batch_z:Tensor, batch_a:Tensor):
        # batch_z: [batch_size, T, hidden_dim]
        # batch_a: [batch_size, T, act_dim]
        batch_z_1 = torch.bmm(self.Kz, batch_z.unsqueeze(-1)).squeeze(-1)
        batch_z_2 = torch.bmm(self.Jz_B, batch_a.unsqueeze(-1)).squeeze(-1)
        batch_z_next = batch_z + (batch_z_1 + batch_z_2)*delta_t
        return batch_z_next

'''
Main Model - Koopman_dynamics
'''

class Koopman_dynamics(nn.Module):
    def __init__(self, state_dim, hidden_dim, act_dim, num_steps, *args, **kwargs) -> None:
        super(Koopman_dynamics, self).__init__(*args, **kwargs)
        # state_dim: dimension of state
        # hidden_dim: dimension of latent space
        # act_dim: dimension of action
        # num_steps: number of steps in recurrent training

        self.num_steps = num_steps
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.act_dim = act_dim

        self.koopman = KoopmanOp(self.hidden_dim, self.act_dim, num_steps=self.num_steps)
        self.encoder = Encoder(self.state_dim, self.hidden_dim)
        self.decoder = Decoder(self.state_dim, self.hidden_dim)
    
    
    def compute_loss(self, s_seq, a_seq, s_next_seq):
        B = s_seq.shape[0]
        device = s_seq.device
        z_seq = torch.zeros(B, self.num_steps, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.num_steps, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        loss_distance_perserving = 0

        for i in range(self.num_steps):
            z_seq[:, i, :] = self.encoder(s_seq[:, i, :])
            z_next_seq[:, i, :] = self.encoder(s_next_seq[:, i, :])

        za_seq = torch.cat((z_seq, a_seq), dim=-1)
        za_seq_pinv = self.batch_pinv(za_seq)


        delta_z = z_next_seq - z_seq
        forward_weights = torch.bmm(za_seq_pinv, delta_z)
        
        self.koopman.Kz = forward_weights[:, :self.hidden_dim, :].transpose(1,2)
        self.koopman.Jz_B = forward_weights[:, self.hidden_dim:, :].transpose(1,2)

        for i in range(self.num_steps):
            pred_z_next = self.koopman.batch_forward(z_seq[:, i, :], a_seq[:, i, :])
            recon_s = self.decoder(z_seq[:, i, :])
            recon_s_next = self.decoder(pred_z_next)

            # predication loss
            loss_fwd = loss_fwd + F.mse_loss(recon_s_next, s_next_seq[:, i, :])
            # identity loss
            loss_identity = loss_identity + F.mse_loss(recon_s, s_seq[:, i, :])
            # distance metric loss
            latent_distance = torch.norm(z_seq[:, i, :] - z_next_seq[:, i, :], p=2, dim=-1)
            original_distance = torch.norm(s_seq[:, i, :] - s_next_seq[:, i, :], p=2, dim=-1)
            loss_distance_perserving = loss_distance_perserving + F.l1_loss(latent_distance, original_distance)

        return loss_fwd, loss_identity, loss_distance_perserving, self.koopman.Kz, self.koopman.Jz_B

    def forward(self, s, a):
        z = self.encoder(s)
        z_next = self.koopman(z, a)
        s_next = self.decoder(z_next)
        return s_next

    def save_model(self, path):
        self.to(cpu_device)
        filename = path + '/' + 'KoopmanDynamics.pt'
        print('[INFO] Saving Koopman Dynamics Model weights to: ', filename)
        torch.save(self.state_dict(), filename)

    @staticmethod
    def batch_pinv(za_seq, I_factor=1e-2):
        """
        cpu implementation only
        :param s_seq: B x T x [(HIDDEN STATE D) + (ACTION D)] (T > [(HIDDEN STATE D) + (ACTION D)])
        :param I_factor: 1e-12
        :return: B x T x [(HIDDEN STATE D) + (ACTION D)]
        """
        B, T, D = za_seq.size()
        device = za_seq.device
        
        # MPS does not support torch.linalg.solve
        if not za_seq.is_cuda:
            za_seq = za_seq.to("cpu")
 
        if T < D:
            za_seq = torch.transpose(za_seq, 1, 2)
            T, D = D, T
            trans = True
        else:
            trans = False
        I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
        za_seq_T = torch.transpose(za_seq, 1, 2)
        za_seq_pinv = torch.linalg.solve(
                      torch.bmm(za_seq_T, za_seq) + I_factor * I, 
                      za_seq_T)
        if trans:
            za_seq_pinv = torch.transpose(za_seq_pinv, 1, 2)
        return za_seq_pinv.to(device)

    def save_Kz_Jz_B(self, path, Kz, Jz_B):
        Kz_filename = path + '/' + 'Kz.pt'
        Jz_B_filename = path + '/' + 'Jz_B.pt'
        print('[INFO] Saving Kz weights to: ', Kz_filename)
        print('[INFO] Saving Jz_B weights to: ', Jz_B_filename)
        torch.save(Kz, Kz_filename)
        torch.save(Jz_B, Jz_B_filename)


class WaveFunction_Koopman_dynamics(nn.Module):
    def __init__(self, pos_dim:int, velocity_dim:int, hidden_dim:int, act_dim:int, num_steps:int, *args, **kwargs) -> None:
        super(WaveFunction_Koopman_dynamics, self).__init__(*args, **kwargs)
        # state_dim: dimension of state
        # hidden_dim: dimension of latent space
        # act_dim: dimension of action
        # num_steps: number of steps in recurrent training

        self.num_steps = num_steps
        self.act_dim = act_dim
        self.pos_dim = pos_dim 
        self.hidden_dim = hidden_dim
        self.velocity_dim = velocity_dim

        self.koopman = KoopmanOp(self.hidden_dim, self.act_dim, num_steps=self.num_steps)
        self.encoder = WaveFunctionEncoder(pos_dim, velocity_dim, hidden_dim)
        self.decoder = WaveFunctionDecoder(pos_dim, velocity_dim, hidden_dim)
    
    
    def compute_loss(self, s_seq, a_seq, s_next_seq):
        B = s_seq.shape[0]
        device = s_seq.device
        z_seq = torch.zeros(B, self.num_steps, self.hidden_dim).to(device)
        z_next_seq = torch.zeros(B, self.num_steps, self.hidden_dim).to(device)

        loss_fwd = 0
        loss_identity = 0
        loss_distance_perserving = 0

        for i in range(self.num_steps):
            z_seq[:, i, :] = self.encoder(s_seq[:, i, :])
            z_next_seq[:, i, :] = self.encoder(s_next_seq[:, i, :])

        za_seq = torch.cat((z_seq, a_seq), dim=-1)
        za_seq_pinv = self.batch_pinv(za_seq)


        delta_z = z_next_seq - z_seq
        forward_weights = torch.bmm(za_seq_pinv, delta_z)
        
        self.koopman.Kz = forward_weights[:, :self.hidden_dim, :].transpose(1,2)
        self.koopman.Jz_B = forward_weights[:, self.hidden_dim:, :].transpose(1,2)

        pred_z_next = self.koopman.batch_forward(z_seq, a_seq)
        
        for i in range(self.num_steps):
            recon_s = self.decoder(z_seq[:, i, :])
            recon_s_next = self.decoder(pred_z_next[:, i, :])
            
            loss_fwd = loss_fwd + F.mse_loss(recon_s_next, s_next_seq[:, i, :])

            # identity loss
            loss_identity = loss_identity + F.mse_loss(recon_s, s_seq[:, i, :])
 
            # distance metric loss
            latent_distance = torch.norm(z_seq[:, i, :] - z_next_seq[:, i, :], p=2, dim=-1)
            original_distance = torch.norm(s_seq[:, i, :] - s_next_seq[:, i, :], p=2, dim=-1)
            loss_distance_perserving = loss_distance_perserving + F.l1_loss(latent_distance, original_distance)

        return loss_fwd, loss_identity, loss_distance_perserving, self.koopman.Kz, self.koopman.Jz_B

    def forward(self, s, a):
        z = self.encoder(s)
        z_next = self.koopman(z, a)
        s_next = self.decoder(z_next)
        return s_next

    def save_model(self, path):
        self.to(cpu_device)
        filename = path + '/' + 'KoopmanDynamics.pt'
        print('[INFO] Saving Koopman Dynamics Model weights to: ', filename)
        torch.save(self.state_dict(), filename)

    @staticmethod
    def batch_pinv(za_seq, I_factor=1e-12):
        """
        cpu implementation only
        :param s_seq: B x T x [(HIDDEN STATE D) + (ACTION D)] (T > [(HIDDEN STATE D) + (ACTION D)])
        :param I_factor: 1e-12
        :return: B x T x [(HIDDEN STATE D) + (ACTION D)]
        """
        B, T, D = za_seq.size()
        device = za_seq.device
        
        za_seq = za_seq.to("cpu")
 
        # assert T > D, "T should be smaller than Hidden Dim"
        if T < D:
            za_seq = torch.transpose(za_seq, 1, 2)
            T, D = D, T
            trans = True
        else:
            trans = False
        I = torch.eye(D)[None, :, :].repeat(B, 1, 1)
        za_seq_T = torch.transpose(za_seq, 1, 2)
        I = I.to("cpu")
        za_seq_pinv = torch.linalg.solve(
                      torch.bmm(za_seq_T, za_seq) + I_factor * I, 
                      za_seq_T)
        if trans:
            za_seq_pinv = torch.transpose(za_seq_pinv, 1, 2)

        return za_seq_pinv.to(device)

    def save_Kz_Jz_B(self, path, Kz, Jz_B):
        Kz_filename = path + '/' + 'Kz.pt'
        Jz_B_filename = path + '/' + 'Jz_B.pt'
        print('[INFO] Saving Kz weights to: ', Kz_filename)
        print('[INFO] Saving Jz_B weights to: ', Jz_B_filename)
        torch.save(Kz, Kz_filename)
        torch.save(Jz_B, Jz_B_filename)

'''
Quadractic Value Network 
'''

class Quad_Value_Net(nn.Module):
    def __init__(self, hidden_dim, *args, **kwargs) -> None:
        super(Quad_Value_Net, self).__init__(*args, **kwargs)
        self.opt_encoded_state = nn.parameter.Parameter(torch.zeros((1, hidden_dim)), 
                                                        requires_grad=False)
        self.hidden_dim = hidden_dim
        self.init_R()

        self.vmap_jacobian = torch.vmap(torch.func.jacrev(self.forward))
    
    def init_R(self):
        R_weight = torch.eye(self.hidden_dim)
        self.R = nn.parameter.Parameter(R_weight, requires_grad=True)
    
    def forward(self, z):
        non_batch = False
        Riemannian = torch.matmul(self.R, self.R.T)
        diff = z - self.opt_encoded_state
        if diff.dim()==1:
            # create the batch dimension
            diff = diff.unsqueeze(0)
            non_batch = True
        # for batch matrix vector multiplication
        diff = diff.unsqueeze(-1)
        value = torch.matmul(torch.matmul(diff.transpose(-1,-2), Riemannian), diff) 
        value = value.squeeze(-1)
        if non_batch:
            value = value.squeeze(0)
        return value
    
    
    def compute_jacobian(self, z):
        out = self.forward(z)
        return torch.autograd.grad(out, z, grad_outputs=torch.ones_like(out), create_graph=True)[0].unsqueeze(0)

    def compute_jacobian_in_batch(self, z):
        J = self.vmap_jacobian(z)
        return J.squeeze(1).requires_grad_(True)

    def save_model(self, path):
        self.to(cpu_device)
        filename = path + '/' + 'Quad_Value_Net.pt'
        print('[INFO] Saving Value Net weights to: ', filename)
        torch.save(self.state_dict(), filename)


'''
Policy: Convex Solver
'''
cpu_device = torch.device('cpu')
class Policy:
    def __init__(self, convex_solver, *args, **kwargs) -> None:
        super(Policy, self).__init__(*args, **kwargs)
        self.solver = convex_solver
    
    @staticmethod
    def obtain_param_in_quadratic_problem_in_batch(koopman_dynamics_model, value_net, 
                                                   states, autoreg=False):
        # Encode the state 
        if autoreg:
            encode_states = states
        else:
            encode_states = koopman_dynamics_model.encoder(states)

        # Compute the Jacobian of the value function w.r.t. latent state z
        value_jocobian_z = value_net.compute_jacobian_in_batch(encode_states)
        value_jocobian_z = value_jocobian_z.squeeze(1).to(cpu_device).detach().numpy()

        # Compute the Jz_B of latent state z
        Jz_B = koopman_dynamics_model.koopman.compute_Jz_B().to(cpu_device).detach().numpy()

        return value_jocobian_z, Jz_B
    
    @staticmethod
    def obtain_param_in_quadratic_problem(koopman_dynamics_model, value_net, states, autoreg=False):
        # Encode the state 
        if autoreg:
            encode_states = states
        else:
            encode_states = koopman_dynamics_model.encoder(states)

        # Compute the Jacobian of the value function w.r.t. latent state z
        value_jocobian_z = value_net.compute_jacobian(encode_states)

        # TODO: take the batch case into account
        value_jocobian_z = value_jocobian_z.to(cpu_device).detach().numpy()

        # Compute the Jz_B of latent state z
        Jz_B = koopman_dynamics_model.koopman.compute_Jz_B().to(cpu_device).detach().numpy()
        
        value_jocobian_z = value_jocobian_z.squeeze(0).squeeze(0)
        Jz_B = Jz_B.squeeze(0)
        return value_jocobian_z.T, Jz_B

    def select_actions(self, koopman_dynamics_model, value_net, states, if_batch=True, autoreg=False):
        if if_batch:        
            value_jocobian_z, Jz_B = \
                self.obtain_param_in_quadratic_problem_in_batch(koopman_dynamics_model, value_net,
                                                                states, autoreg)
            return self.solver.solve_in_batch(value_jocobian_z, Jz_B)
        else:
            value_jocobian_z, Jz_B = \
                self.obtain_param_in_quadratic_problem(koopman_dynamics_model, value_net,
                                                       states, autoreg)
            opt_action = self.solver.solve(value_jocobian_z, Jz_B)
            return opt_action
        

