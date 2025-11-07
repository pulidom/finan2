#import torch
#import numpy as np
import nn.lossfn as lossfn

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device='cpu'

seed=2
exp_dir= './dat/finan'
sexp = 'regime-breaks'

# Define parametros de los datos
class dat:
    pathdat='dat/'
    tipo='asset'#log' #'asset' # 'asset', 'return', 'log_return', 'log'
    n_train = 50_000
    n_val = 5_000
    Njump = 10 # peque~no para el entrenamiento
    beta_win=121 #*121   #21
    zscore_win=41 #11
    batch_size=256
    nt_start=beta_win
    nt_window=beta_win+zscore_win
    fname=f'tmp/syn_pair_{sexp}_' # fig filename
    shorten=0
        
# Define parametros de la optimizacion
class train: 
    loss = lossfn.loss_fn #nn.MSELoss() #nn.logGaussLoss #SupervLoss # GaussLoss
    batch_size=dat.batch_size
    n_epochs = 100
    learning_rate = 1e-3
    exp_dir = exp_dir
    lvalidation = True 
    patience = 30
    sexp = sexp
    
# Define parametros de la red
class net:
    hidden_dim=40
    layers=3
    input_dim=2 #[0,1]#len(dat.var_in)
    output_dim=1#[0] #len(dat.var_out)
    dropout=0.1
    device=device
    nt_start=100 # warm-up times/ input_times
    nt_window=dat.nt_window
    n_samples = 100

