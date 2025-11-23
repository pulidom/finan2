import numpy as np
#import matplotlib.pyplot as plt
import my_pyplot as plt
import torch, os, copy
from torch.utils import data 
from torch.autograd import Variable
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from statsmodels.nonparametric.kernel_regression import KernelReg
from numpy.polynomial import polynomial as P
from scipy import stats
#import optimal_transport as ot
import ot3
from time import time

def NLL_loss(pred, labels, eps=1e-6):
    ''' Negative log likelihood '''
    mu, sigma = pred.T
    mu=mu[:,None]; sigma=sigma[:,None]
    sigma = torch.clamp(sigma, min=eps)
    distribution = torch.distributions.normal.Normal(mu, sigma)
    return -torch.mean(distribution.log_prob(labels))

def MSE_loss(pred, labels, epoch):
     mu, sigma = pred.T
     return F.mse_loss(mu, labels.squeeze())

def mixed_loss(pred, labels, epoch, warmup_epochs=30):
    """Gradually transition from MSE to NLL"""
    mu, sigma = pred.T
    
    # Warmup: use more MSE early on
    alpha = min(1.0, epoch / warmup_epochs)
    
    # MSE component
    mse = F.mse_loss(mu, labels.squeeze())
    
    # NLL component
    sigma = torch.clamp(sigma, min=1e-6)
    nll = torch.mean(0.5 * ((labels.squeeze() - mu) / sigma) ** 2 + torch.log(sigma))
    
    return (1 - alpha) * mse + alpha * nll


def eMSE_loss( in_var,labels,alpha=0.5):
    """
    Extended MSE using estimated variance
    """
    mu,var = in_var.T
    mu=mu[:,None]; var=var[:,None]
    error = (mu-labels)**2
    return (1-alpha) * F.mse_loss(var, error)+alpha*F.mse_loss(mu, labels)

def boundary_loss(in_var, labels, epoch, warmup_epochs=20, x_data=None, 
                                  boundary_targets=None, boundary_penalty_weight=10.0):
    """
    Loss with smooth boundary guidance
    """
    mu, sigma = in_var.T
    labels = labels.squeeze()
    
    # Standard warmup
    alpha = min(1.0, epoch / warmup_epochs)
    mse = F.mse_loss(mu, labels)
    
    sigma = torch.clamp(sigma, min=1e-6)
    nll = torch.mean(0.5 * ((labels - mu) / sigma) ** 2 + torch.log(sigma))
    
    base_loss = (1 - alpha) * mse + alpha * nll
    
    # Smooth boundary penalty with distance weighting
    if x_data is not None and boundary_targets is not None:
        x = x_data.squeeze()
        y_left, y_right = boundary_targets
        
        # Distance-based weights (exponentially decay from boundaries)
        # Closer to boundary = higher weight
        left_weight = torch.exp(-10 * x)  # High at x=0, decays quickly
        right_weight = torch.exp(-10 * (1 - x))  # High at x=1, decays quickly
        
        # Boundary errors weighted by distance
        left_boundary_loss = (left_weight * (mu - y_left) ** 2).mean()
        right_boundary_loss = (right_weight * (mu - y_right) ** 2).mean()
        
        # Add to loss with scheduling (start after some warmup)
        if epoch > warmup_epochs // 2:
            boundary_weight = boundary_penalty_weight * min(1.0, (epoch - warmup_epochs//2) / warmup_epochs)
            base_loss = base_loss + boundary_weight * (left_boundary_loss + right_boundary_loss)
    
    return base_loss

def set_seed(seed=42):
    """Set seed for reproducibility"""
    # Python random
    
    # Numpy
    np.random.seed(seed)
    
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    # PyTorch backends
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

class BoundaryNet(nn.Module):
    def __init__(self, base_net, boundary_width=0.15, min_sigma=0.02):
        super().__init__()
        self.base_net = base_net
        self.boundary_width = boundary_width
        self.min_sigma = min_sigma
    
    def forward(self, x):
        pred = self.base_net(x)
        mu, sigma = pred[:, 0:1], pred[:, 1:2]
        
        # Distance from nearest boundary
        x_val = x.squeeze()
        dist_from_boundary = torch.minimum(x_val, 1.0 - x_val)
        
        # Smooth transition: 0 at boundary, 1 far from boundary
        # Using sigmoid for smooth interpolation
        transition = torch.sigmoid((dist_from_boundary - self.boundary_width/2) * 20)
        transition = transition.unsqueeze(1)
        
        # Interpolate between min_sigma and network prediction
        sigma_adjusted = self.min_sigma + (sigma - self.min_sigma) * transition
        
        return torch.cat([mu, sigma_adjusted], dim=1)

#
#def get_polynomial_boundaries(train_x, train_y, degree=5):
#    """Fit polynomial and extract boundary predictions"""
#    # Fit polynomial
#    coefs = P.polyfit(train_x, train_y, degree)
#    
#    # Evaluate at boundaries
#    y_at_0 = P.polyval(0.0, coefs)
#    y_at_1 = P.polyval(1.0, coefs)
#    
#    return y_at_0, y_at_1, coefs
#
#def evaluate_polynomial(x, coefs):
#    """Evaluate polynomial at x"""
#    return P.polyval(x, coefs)
#
## Get polynomial fit from training data
#train_x, train_y = xydat[0]
#y_left, y_right, poly_coefs = get_polynomial_boundaries(train_x, train_y, degree=6)
#
#print(f"Polynomial boundary values: left={y_left:.3f}, right={y_right:.3f}")
#
# Now use these as hard constraints
class PolynomialBoundaryNet(nn.Module):
    def __init__(self, base_net, y_left, y_right, boundary_width=0.1):
        """
        Enforce that predictions match polynomial at boundaries
        y_left: target value at x=0
        y_right: target value at x=1
        """
        super().__init__()
        self.base_net = base_net
        self.y_left = torch.tensor(y_left, dtype=torch.float32)
        self.y_right = torch.tensor(y_right, dtype=torch.float32)
        self.boundary_width = boundary_width
    
    def forward(self, x):
        pred = self.base_net(x)
        mu, sigma = pred[:, 0:1], pred[:, 1:2]
        
        x_val = x.squeeze()
        
        # Weight functions: 1 at boundary, 0 in middle
        w_left = torch.exp(-((x_val - 0.0) / self.boundary_width) ** 2)
        w_right = torch.exp(-((x_val - 1.0) / self.boundary_width) ** 2)
        
        # Blend network prediction with polynomial boundary values
        y_left_dev = self.y_left.to(x.device)
        y_right_dev = self.y_right.to(x.device)
        
        mu_corrected = (mu.squeeze() * (1 - w_left - w_right) + 
                       y_left_dev * w_left + 
                       y_right_dev * w_right).unsqueeze(1)
        
        # Also reduce sigma at boundaries
        min_sigma = 0.01
        sigma_scale = 1.0 - 0.95 * (w_left + w_right).unsqueeze(1)
        sigma_adjusted = min_sigma + sigma * sigma_scale
        
        return torch.cat([mu_corrected, sigma_adjusted], dim=1)
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)

def NNmodel(layers,activation_fn,activation_output=None):
    """ 
    Generate an NN with linear transformations and activation functions 
    """
    # Parameter Initialization with .apply
    modules = []
    for i in range(len(layers)-2):
        modules.append(nn.Linear(layers[i], layers[i+1]))
        modules.append(activation_fn)
    #output layer
    modules.append(nn.Linear(layers[len(layers)-2], layers[len(layers)-1]))
    if activation_output is not None:
        modules.append(activation_output)

    NNmdl = nn.Sequential(*modules)
    # aca puedo inicializar la red
    #NNmdl.apply(init)
    
    return NNmdl


#class ResidualBlock(nn.Module):
#    def __init__(self, dim, activation):
#        super().__init__()
#        self.linear1 = nn.Linear(dim, dim)
#        self.linear2 = nn.Linear(dim, dim)
#        self.activation = activation
#    
#    def forward(self, x):
#        residual = x
#        out = self.activation(self.linear1(x))
#        out = self.linear2(out)
#        return self.activation(out + residual)  # Skip connection
#
#class ResNet(nn.Module):
#    def __init__(self, input_dim, hidden_dim, output_dim, n_blocks, activation):
#        super().__init__()
#        self.input_layer = nn.Linear(input_dim, hidden_dim)
#        self.blocks = nn.ModuleList([
#            ResidualBlock(hidden_dim, activation) for _ in range(n_blocks)
#        ])
#        self.output_layer = nn.Linear(hidden_dim, output_dim)
#        self.activation = activation
#    
#    def forward(self, x):
#        x = self.activation(self.input_layer(x))
#        for block in self.blocks:
#            x = block(x)
#        return self.output_layer(x)

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

    t0=time()
    for i_epoch in range(conf.n_epochs):
        # parte de entrenamiento
        print (i_epoch,'de', conf.n_epochs)
        Net.train() 
        train_loss = 0
        
        #beta= get_beta(i_epoch, warmup=2, total=12, beta_min=0.2, dbeta_max=0.3)
        #beta=0.5
        for i_batch, (input_dat,target_dat) in enumerate(train_loader):

            optimizer.zero_grad()
            
            #prediction = Net.train_step(input_dat)
            prediction = Net(input_dat)
            cost = loss( prediction, target_dat, i_epoch)#,beta=beta)
#            beta= get_beta(prediction, target_dat)

            train_loss += cost.item()
            cost.backward()
            #print("grad",next(Net.parameters()).grad.abs().mean().item())
            optimizer.step()

        loss_t[i_epoch]=train_loss/(i_batch+1)
        print('cost train',loss_t[i_epoch])
        print('time: ',time()-t0)
        # Validation
        if conf.lvalidation and i_epoch % freq_epoch == 0: 
            Net.eval() 
            with torch.no_grad():
                val_loss=0
                for i_batch,  (input_dat,target_dat) in enumerate(val_loader):
                    #prediction = Net.train_step(input_dat)
                    prediction = Net(input_dat)
                    #print('spread: ',prediction.std(dim=1))
                    cost = loss( prediction, target_dat, i_epoch )
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
    np.savez(conf.exp_dir+f'/rmses_{conf.sexp}.npz', losstrain=loss_t,lossval=loss_v)
    if best_net is None:
        model_file=os.path.join(conf.exp_dir, f'model_{conf.sexp}_last.ckpt')
        torch.save(Net,model_file)
        best_net=Net
        

    return best_net,loss_t,loss_v

class DriveData(Dataset):
    "para utilizarse con el DataLoader de pytorch "
    def __init__(self,dat,n_samples=10000,device='cpu',normalize=False):

        self.xs,self.ys=dat #Sampler(n_samples)

        self.xs=self.xs[:,None]
        self.ys=self.ys[:,None]
        
        self.x_data = torch.from_numpy(np.asarray(self.xs,dtype=np.float32)).to(device)
        self.y_data = torch.from_numpy(np.asarray(self.ys,dtype=np.float32)).to(device)

        if normalize is not None:
            Norm_x=Normalizator(self.x_data,tipo=normalize)
            Norm_y=Normalizator(self.y_data,tipo=normalize)
            self.x_data = Norm_x.normalize(self.x_data)
            self.y_data = Norm_y.normalize(self.y_data)
        
        self.lenx=self.xs.shape[0]
        
    def __getitem__(self,index):
        return self.x_data[index], self.y_data[index]    
    def __len__(self):
        return self.lenx
    def getnp(self):
        return self.xs, self.ys

class Normalizator(nn.Module):
    def __init__(self, X, tipo='gauss'):
        super().__init__()
        self.tipo = tipo
        
        # Registrar buffers # si los quiero guardar en la red
        #self.register_buffer('mean', None)
        #self.register_buffer('std', None)
        #self.register_buffer('min_val', None)
        #self.register_buffer('max_val', None)
        
        if self.tipo == 'gauss':
            self.normalize = self.normalize_gauss
            self.desnormalize = self.desnormalize_gauss
            self.media = torch.mean(X, dim=(0))
            self.std = torch.std(X, dim=(0))
        elif self.tipo == 'minmax':
            self.normalize = self.normalize_minmax
            self.desnormalize = self.desnormalize_minmax
            self.min_val = torch.min(X, dim=(0))[0]
            self.max_val = torch.max(X, dim=(0))[0]
    
    def normalize_gauss(self, X):
        return (X - self.media) / self.std
    
    def desnormalize_gauss(self, Xnorm):
        return self.media + self.std * Xnorm
        
    def normalize_minmax(self, X):
        return (X - self.min_val) / (self.max_val - self.min_val)
    
    def desnormalize_minmax(self, Xnorm):
        return self.min_val + (self.max_val - self.min_val) * Xnorm  # ← Xnorm corregido

def sin_sampler(n_samples):
    x = np.random.random(n_samples)
    eps = 0.2 * np.random.normal(scale=1.0, size=n_samples) * (1- 0.6*np.cos(x*2*np.pi))
    y = 2*np.sin(2*np.pi*x) + eps
    return x,y


def syndata_polynomial(real_x, real_y, n_synthetic, degree=6, noise_scale=1.0):
    """
    Generate synthetic data by:
    1. Fitting polynomial to real data
    2. Sampling from polynomial + noise

       """
    
    # Fit polynomial to real data
    poly_coefs = P.polyfit(real_x, real_y, degree)
    
    # Estimate noise level from real data residuals
    y_pred_real = P.polyval(real_x, poly_coefs)
    residuals = real_y - y_pred_real
    print(min(residuals),max(residuals))
    
    noise_std = np.std(residuals)
    
    # Generate synthetic data
    synthetic_x = np.random.uniform(real_x.min(), real_x.max(), n_synthetic)
    
    # Polynomial predictions
    y_poly = P.polyval(synthetic_x, poly_coefs)
    
    # Add noise with estimated scale
    noise_stds =  noise_std * noise_scale * np.cos(np.pi * (synthetic_x - 0.5))
    noise = noise_stds * np.random.normal(0,1 , n_synthetic)

    synthetic_y = y_poly + noise
    
    return synthetic_x, synthetic_y, poly_coefs

