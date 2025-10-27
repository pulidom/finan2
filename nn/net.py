'''Defines the recursive neural network
    Version preliminar tomada de deepar
     Se la piensa para sistemas dinamicos. 
     Se la hace mas eficiente.
     Se maneja teacher forcing y prediccion desde la propia clase Net
     Generalizo a multiples outputs
     Covariates are inputs without ouput (at the last part of the input tensor)

'''

import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from nn.LSTM import LSTM # dropout from Gal & Ghahramani
#import torch.nn.functional as F
#from torch.autograd import Variable

class Net(nn.Module):
    
    def __init__(self, conf):
        '''
        We define a recurrent network that predicts the future values 
        of a time-dependent variable based on
        past inputs and covariates.
        '''
        super().__init__()
        self.conf = conf
        
        # Seleccionar LSTM o GRU
        rnn_type = getattr(conf, 'rnn_type', 'LSTM')  # default LSTM
        
        if rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=conf.input_dim,
                               hidden_size=conf.hidden_dim,
                               num_layers=conf.layers,
                               bias=True,
                               batch_first=False,
                               dropout=conf.dropout)
            self.is_lstm = False
        else:  # LSTM
            self.rnn = nn.LSTM(input_size=conf.input_dim,
                                hidden_size=conf.hidden_dim,
                                num_layers=conf.layers,
                                bias=True,
                                batch_first=False,
                                dropout=conf.dropout)
            self.is_lstm = True
        
        self.initialize_forget_gate() 
        self.return_state = True #conf.return_state
        self.output_dim = conf.output_dim

        # Capas finales transforman: hidden_dim -> output_dim 
        self.FC_mu = nn.Linear(conf.hidden_dim, conf.output_dim)
        self.FC_sigma = nn.Sequential(
            nn.Linear(conf.hidden_dim, conf.output_dim),
            nn.Softplus(),) # softplus to make sure standard deviation is positive
        
        #self.FC_mu = nn.Linear(conf.hidden_dim * conf.layers, conf.output_dim)
        #self.FC_sigma = nn.Sequential(
        #    nn.Linear(conf.hidden_dim * conf.layers, conf.output_dim),
        #    nn.Softplus(),) # softplus to make sure standard deviation is positive

    def forward(self, input, hidden_stt=None):
        '''
        Predict mu and sigma of the distribution for z_t.
        Args:
             x: ([1, batch_size, input_dim]): z_{t-1} + x_t, note that z_0 = 0
            hidden_stt: (hidden, cell) for LSTM or hidden for GRU
        Returns:
            mu ([batch_size]): estimated mean of z_t
            sigma ([batch_size]): estimated standard deviation of z_t
            hidden_stt: (hidden, cell) for LSTM or hidden for GRU
        '''
        
        output, hidden_stt = self.rnn(input, hidden_stt)

        # use hidden stt from all the LSTM layers to calculate mu and sigma
        #hidden_permute = hidden.permute(1, 2, 0).contiguous().view(hidden.shape[1], -1)
        #mu = self.FC_mu(hidden_permute)
        #sigma = self.FC_sigma(hidden_permute)

        # Directly from ouptput of LSTM
        mu = self.FC_mu(output)
        sigma = self.FC_sigma(output)

        outputs = mu, sigma # torch.squeeze(mu), torch.squeeze(sigma)

        #mu and sigma shape [batch_size, conf.output_dim]
        if self.return_state:
            return outputs, hidden_stt
        else:
            return outputs

    def init_hidden(self, input_size):
        hidden = torch.zeros(self.conf.layers, input_size,
                           self.conf.hidden_dim, device=self.conf.device)
        if self.is_lstm:
            cell = torch.zeros(self.conf.layers, input_size,
                             self.conf.hidden_dim, device=self.conf.device)
            return (hidden, cell)
        else:
            return hidden

    def init_cell(self, input_size):
        return torch.zeros(self.conf.layers, input_size,
                           self.conf.hidden_dim, device=self.conf.device)
    
    def initialize_forget_gate(self):
        ''' initialize LSTM forget gate bias to be 1 as recommended
        by http://proceedings.mlr.press/v37/jozefowicz15.pdf
        '''
        if self.is_lstm:
            for names in self.rnn._all_weights:
                for name in filter(lambda n: "bias" in n, names):
                    bias = getattr(self.rnn, name)
                    n = bias.size(0)
                    start, end = n // 4, n // 2
                    bias.data[start:end].fill_(1.)  # Forget gate = 1

    def train_step(self, train_batch):
        '''
        Para el entrenamiento forward pass in the training time window 
        con teacher forcing.
        
        '''
        batch_size = train_batch.shape[1]
        seq_len = train_batch.shape[0]
        hidden_stt = self.init_hidden(batch_size)

        self.return_state = True
        mu_t = torch.zeros(seq_len, batch_size, self.output_dim, device=self.conf.device)
        sigma_t = torch.zeros(seq_len, batch_size, self.output_dim, device=self.conf.device)

        for t in range(seq_len):

            (mu_t[t], sigma_t[t]), hidden_stt = self.forward(
                train_batch[t].unsqueeze_(0).clone(), 
                hidden_stt
            )

        return mu_t, sigma_t
    
    def pred(self, x, deterministic=True):
        ''' 
         Prediccion usando warmup time y luego si autoregresivo
           option deterministica o estocastica
          x = [seq,batch,output_dim]
        '''

        batch_size = x.shape[1]
        nt_start = self.conf.nt_start
        nt_window = self.conf.nt_window
        nt_pred = nt_window - self.conf.nt_start
        n_samples = self.conf.n_samples
        
        hidden_stt = self.init_hidden(batch_size)

        mu = torch.zeros(nt_window, batch_size, self.output_dim, device=self.conf.device)
        sigma = torch.zeros(nt_window, batch_size, self.output_dim, device=self.conf.device)
        
        self.eval()
        with torch.no_grad():

            # warming up period
            for t in range(nt_start):
                (mu[t], sigma[t]), hidden_stt = self.forward(
                    x[t].unsqueeze_(0).clone(), 
                    hidden_stt)

            # prediction period: 
            if deterministic:
                for t in range(nt_start,nt_window):
                    xaug=torch.cat([mu[t-1],x[t-1,...,self.output_dim:]],dim=-1)
                    (mu[t], sigma[t]), hidden_stt = self.forward(
                        xaug.unsqueeze_(0).clone(), 
                        hidden_stt)
                return mu, sigma,  0
            else: # stochastic
                samples = torch.zeros(
                    nt_pred, n_samples,batch_size, 
                    self.output_dim,
                    device=self.conf.device
                )
                # replico para todas las muestras el hidden state
                if self.is_lstm:
                    hidden, cell = hidden_stt
                    hidden = hidden.repeat_interleave(n_samples, dim=1)
                    cell = cell.repeat_interleave(n_samples, dim=1)
                    hidden_stt = (hidden, cell)
                else:
                    hidden_stt = hidden_stt.repeat_interleave(n_samples, dim=1)

                # Inicializo con replicas [n_samples, batch_size, output_dim]
                mu_rep = mu[nt_start-1].unsqueeze(0).repeat(n_samples, 1, 1)  
                sigma_rep = sigma[nt_start-1].unsqueeze(0).repeat(n_samples, 1, 1) 

                gaussian = Normal(mu_rep, sigma_rep)
                sampler = gaussian.sample().view(n_samples * batch_size, self.output_dim)

                # Prediction with Autoregression 
                for t in range(nt_start, nt_window):
                    covariates = x[t, :, self.output_dim:].repeat_interleave(n_samples, dim=0)
                    xaug = torch.cat([sampler, covariates], dim=-1)

                    (mu_samples, sigma_samples), hidden_stt = self.forward(
                        xaug.unsqueeze(0).clone(),
                        hidden_stt
                    )

                    gaussian = Normal(mu_samples.squeeze(0), sigma_samples.squeeze(0))
                    sampler = gaussian.sample()

                    samples[t-nt_start] = sampler.view(n_samples, batch_size, self.output_dim)
                    mu[t] = torch.median(samples[t - nt_start], dim=0)[0]
                    sigma[t] = samples[t - nt_start].std(dim=0)
                # samples [n_sample, nt_window-nt_start,batch_size,output_dim]
                return mu, sigma, samples
