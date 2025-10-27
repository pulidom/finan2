import os
import copy
import torch
import numpy as np

def train(Net,train_loader,val_loader,conf):
    '''   Entrena una red neuronal dados los datos
             conf: loss
                   lr
                   n_epochs
                   patience
                   save_loss
          Asumo una funcion lo mas transparente posible para que
         pueda ser usada en diferentes proyectos.
          [2025-09-27] 
    ''' 

    
    loss=conf.loss
    optimizer = torch.optim.Adam(Net.parameters(), lr=conf.learning_rate)   

    best_mse = np.inf
    best_net = None
    freq_epoch = getattr(conf, 'freq_epoch', 1)
    loss_t,loss_v=np.inf * np.ones((2,conf.n_epochs))
    
    for i_epoch in range(conf.n_epochs):
        # parte de entrenamiento
        print (i_epoch,'de', conf.n_epochs)
        Net.train() 
        train_loss = 0
        for i_batch, (input_dat,target_dat) in enumerate(train_loader):

            optimizer.zero_grad()
            input_dat=input_dat.transpose(0,1)
            target_dat=target_dat.transpose(0,1)
            
            prediction = Net.train_step(input_dat)
            cost = loss( prediction, target_dat)

            train_loss += cost.item()
            cost.backward()
            optimizer.step()

        loss_t[i_epoch]=train_loss/(i_batch+1)
        print('cost train',loss_t[i_epoch])

        # Validation
        if conf.lvalidation and i_epoch % freq_epoch == 0: 
            Net.eval() 
            with torch.no_grad():
                val_loss=0
                for i_batch,  (input_dat,target_dat) in enumerate(val_loader):
                    input_dat=input_dat.transpose(0,1)
                    target_dat=target_dat.transpose(0,1)
                    prediction = Net.train_step(input_dat)
                    cost = loss( prediction, target_dat )
                    val_loss += cost.item()

                loss_v[i_epoch]=val_loss/(i_batch+1)
                print('cost val: ',loss_v[i_epoch])

                if loss_v[i_epoch] < best_mse:
                    best_mse = loss_v[i_epoch]
                    tolerate_iter = 0 
                    model_file=os.path.join(conf.exp_dir, f'model_{conf.sexp}_best.ckpt')
                    torch.save(Net,model_file)
                    best_net = copy.deepcopy(Net)

                else:
                    tolerate_iter += 1
                    if tolerate_iter == conf.patience:
                        print('the best validation loss is:', best_mse)
                        break

    #----------->
    np.savez(conf.exp_dir+f'/rmses.npz', losstrain=loss_t,lossval=loss_v)
    if best_net is None:
        model_file=os.path.join(conf.exp_dir, f'model_{conf.sexp}_last.ckpt')
        torch.save(Net,model_file)
        best_net=Net
        

    return best_net,loss_t,loss_v

def test(test_loader,Net,deterministic=True):
    ''' Produces a prediction assuming I have a single batch '''
    with torch.no_grad():
        for i_batch,  (input_dat,target_dat) in enumerate(test_loader):
            print(input_dat.shape)
            input_dat=input_dat.transpose(0,1)
            target_dat=target_dat.transpose(0,1)
 
            mu,sigma,sample = Net.pred(input_dat,deterministic=True)
    # conversion a numpy?
    return mu.cpu().detach().numpy(),sigma.cpu().detach().numpy(),target_dat.cpu().detach().numpy()
