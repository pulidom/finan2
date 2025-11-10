''' Entrena una red neuronal fully conected para inferir la media y la varianza dado un input (activo y)
    y un output (activo x)
'''


import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn.functional as F


def emse_loss(input, target):
    
    # Estimate target value for variance (sigma^2) with (y_pred - y)**2
    #
    #    actual y        is target[:,0]
    # predicted y        is input[:,0]
    #    actual variance is target[:,1] - estimated here
    # predicted variance is input[:,0]
    
    # Use 'requires_grad == False' to prevent PyTorch from trying to differentiate 'target'
    target[:,1] = Variable((input[:,0].data - target[:,0].data)**2, 
                           requires_grad=False)  
    # Return MSE loss 
    return F.mse_loss(input, target)


def train(x_train,y_train,x_val,y_val,device,loss):
          
    ## Mando los 

    x_train = torch.from_numpy(np.asarray(x_train,dtype=np.float32)).to(device)
    y_train = torch.from_numpy(np.asarray(y_train,dtype=np.float32)).to(device)


    # Create a simple two-layer network with one input (x) and two outputs (y, sigma)
    n_inputs = 1
    n_outputs = 2
    n_hidden = 1000
    model = torch.nn.Sequential(torch.nn.Linear(n_inputs, n_hidden),
                                torch.nn.ReLU(),
                                torch.nn.Linear(n_hidden, n_outputs)
                               ).to(device)

    # Adam optimizer
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    n_epochs = 10000 
    for i in range(n_epochs):

        # Calculate predicted y from x
        y_pred = model(x_train) ### with the whole batch!!!!

        # Calculate loss
        loss_t = loss(y_pred, y_train)

        # Backprop, first zeroing gradients
        optimizer.zero_grad()
        loss.backward()

        # Update parameters
        optimizer.step()
        if i%100 == 0:

            with torch.no_grad():
                y_pred = model(x_val) ### with the whole batch!!!!

                # Calculate loss
                loss_v = loss(y_pred, y_val)
                
                print(f'epoch: {i:4}, loss train: {loss_t.value:.3}',)
                print(f'epoch: {i:4}, loss valid: {loss_v.value:.3}',)
        
    return model
