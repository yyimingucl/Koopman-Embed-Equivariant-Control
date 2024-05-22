'''
Filename: train.py
Created: 24/01/2024
Description:
    This file contains the training functions.
'''

import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

def lr_scheduler(optimizer, lr_decay_rate=0.5, min_lr=1e-8):
    """Decay learning rate by a factor of lr_decay_rate every lr_decay_epoch epochs"""
    for param_group in optimizer.param_groups:
        if param_group['lr'] > min_lr:
            param_group['lr'] *= lr_decay_rate
    return optimizer

def train_koopman_dynamics(model, train_dataset, train_logger, model_save_path, 
                           num_epochs=30, batch_size=64, 
                           lr=1e-3, lr_decay=0.5, lr_decay_step=5,
                           device="cpu", gradclip=1, 
                           lamb=1, beta=1e-1, 
                           evaluate=False, test_dataset=None, test_logger=None):
    print('[INFO] Start Training Koopman Dynamics Model')
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    num_samples = len(train_dataset)
    if evaluate:
        assert test_dataset is not None, "[WARNING] Test dataset is not provided"
        assert test_logger is not None, "[WARNING] Test logger is not provided"

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        test_loss = {'forward':[[] for _ in range(num_epochs)], 'autoencoder':[[] for _ in range(num_epochs)], 
                     'latent_forward':[[] for _ in range(num_epochs)], 'distance_preserving':[[] for _ in range(num_epochs)]}
        

    train_loss = {'forward':[[] for _ in range(num_epochs)], 'autoencoder':[[] for _ in range(num_epochs)], 
                  'latent_forward':[[] for _ in range(num_epochs)], 'distance_preserving':[[] for _ in range(num_epochs)]}
    
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(trainable_parameters, lr=lr)
    
    model.train()       
    model.to(device)

    for epoch in range(num_epochs):
        Kz = torch.zeros((model.hidden_dim, model.hidden_dim), dtype=torch.float32).to(device)
        Jz_B = torch.zeros((model.hidden_dim, model.act_dim), dtype=torch.float32).to(device)
        for _, batch_data in tqdm(enumerate(train_dataloader)):
            
            seq_state, seq_actions, following_state, _, _ = [each_data.to(device) for each_data in batch_data]

            loss_fwd, loss_identity, loss_distance_perserving, tmp_Kz, tmp_Jz_B = model.compute_loss(seq_state, seq_actions, following_state)

            loss = loss_fwd + lamb*loss_identity + beta*loss_distance_perserving
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, gradclip) # gradient clip
            optimizer.step()      

            train_loss['forward'][epoch].append(loss_fwd.cpu().detach().numpy())
            train_loss['autoencoder'][epoch].append(loss_identity.cpu().detach().numpy())
            train_loss['distance_preserving'][epoch].append(loss_distance_perserving.cpu().detach().numpy())
            Kz += tmp_Kz.mean(dim=0)*(batch_size/num_samples)
            Jz_B += tmp_Jz_B.mean(dim=0)*(batch_size/num_samples)

        epoch_fwd_loss = np.mean(train_loss['forward'][epoch])
        epoch_identity_loss = np.mean(train_loss['autoencoder'][epoch])
        epoch_distance_preserving_loss = np.mean(train_loss['distance_preserving'][epoch])

        train_logger.log_metrics(epoch, loss_forward = epoch_fwd_loss,
                                        loss_identity = epoch_identity_loss,
                                        loss_distance_preserving = epoch_distance_preserving_loss)
    


        model.save_model(model_save_path)
        model.save_Kz_Jz_B(model_save_path, Kz, Jz_B)
        model.to(device)
        
        if (epoch+1) % lr_decay_step == 0:
            print('[INFO] Learning Rate Decay')
            optimizer = lr_scheduler(optimizer, lr_decay_rate=lr_decay)

        if evaluate:
            with torch.no_grad():
                for _, batch_data in enumerate(test_dataloader):
                
                    seq_state, seq_actions, following_state, _, _ = batch_data.to(device)
                    loss_fwd, loss_identity, loss_distance_perserving = model.compute_loss(seq_state, seq_actions, following_state)
  
                    test_loss['forward'][epoch].append(loss_fwd.cpu().detach().numpy())
                    test_loss['autoencoder'][epoch].append(loss_identity.cpu().detach().numpy())
                    test_loss['distance_preserving'][epoch].append(loss_distance_perserving.cpu().detach().numpy())
        
            test_logger.log_metrics(epoch, loss_forward=np.mean(test_loss['forward'][epoch]),
                                           loss_identity=np.mean(test_loss['autoencoder'][epoch]),
                                           loss_distance_preserving=np.mean(test_loss['distance_preserving'][epoch]))
        
    return train_loss, test_loss if evaluate else train_loss



def train_value_net(value_net, koopman_dynamics_model, train_dataset, number_steps, 
                    logger, save_model_path, device, 
                    lr=1e-5, lr_decay_rate=0.5, lr_decay_step=5, 
                    weight_decay=0, gamma=0.99, num_epochs=10, batch_size=64, gradclip=1.):
    # gamma is the discount factor
    assert gamma <= 1 and gamma >= 0, 'gamma should be in [0,1]'

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)     
    
    optimizer = torch.optim.AdamW(value_net.parameters(), lr=lr, weight_decay=weight_decay)  

    criterion = nn.MSELoss()
    train_loss = {'td_error':[[] for _ in range(num_epochs)]}

    print('[INFO] Freeze Koopman Dynamics Model')
    for param in koopman_dynamics_model.parameters():
        param.requires_grad = False
    koopman_dynamics_model.to(device)
    value_net.to(device)

    for epoch in range(num_epochs):
        # Training
        value_net.train()
        for _, batch_data in tqdm(enumerate(train_dataloader)):          
            seq_state, _, following_state, rewards, _ = [each_data.to(device) for each_data in batch_data]
            loss = 0
            for i in range(number_steps):
                z = koopman_dynamics_model.encoder(seq_state[:,i,:].clone())

                z_next = koopman_dynamics_model.encoder(following_state[:,i,:].clone())
                reward = rewards[:,i].unsqueeze(1)
                diff = value_net(z) - gamma*value_net(z_next)

                loss = loss+criterion(diff, reward)
               

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), gradclip)
            optimizer.step()

            train_loss['td_error'][epoch].append(loss.cpu().detach().numpy())

        scheduler_loss = np.mean(train_loss['td_error'][epoch])
        logger.log_metrics(epoch, td_error=scheduler_loss)

        # Save the model
        value_net.save_model(save_model_path)
        value_net.to(device)
        
        if (epoch+1) % lr_decay_step == 0:
            optimizer = lr_scheduler(optimizer, lr_decay_rate)

    print('[INFO] Defreeze Koopman Dynamics Model')
    for param in koopman_dynamics_model.parameters():
        param.requires_grad = True
    return train_loss

