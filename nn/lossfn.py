import torch


def loss_fn(in_var, labels):
    '''
    Compute using gaussian the log-likehood which needs to be maximized. Ignore time steps where labels are missing.
    Args:
       in_var composed of:
        mu: (Variable) dimension [batch_size] - estimated mean at time step t
        sigma: (Variable) dimension [batch_size] - estimated standard deviation at time step t
        labels: (Variable) dimension [batch_size] z_t
    Returns:
        loss: (Variable) average log-likelihood loss across the batch
    Tomada de deepAR
    '''
    mu,sigma = in_var
    distribution = torch.distributions.normal.Normal(mu, sigma)
    likelihood = distribution.log_prob(labels)
    return -torch.mean(likelihood)

def emse_var( in_var,labels):
    " Extended MSE with estimated variance"
    mu,sigma = in_var
    # falta definir si es sigma o la varianza
    var = tor.pow(tor.sub(mu,labels),2)
    return F.mse_loss(sigma, var)
