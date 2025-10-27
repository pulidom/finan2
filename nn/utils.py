import torch
from torch.distributions import Normal
import math
EPS=1.e-9

def leadtime_rmse(target,pred):
    ''' rmse en fn del lead time [seq,batch,output_dim]'''
    return ((((pred-target)**2).mean(-1))**0.5).mean(1)

def crps(y_true, y_samples):
    """
    y_true: [seq, batch, output_dim]
    y_samples: [seq, batch, n_samples, output_dim]
    returns: crps_seq [seq] averaged over batch & output_dim
    """
    seq, batch, n_samples, out_dim = y_samples.shape

    # Expand true values for broadcasting
    y_true_exp = y_true.unsqueeze(2)  # [seq, batch, 1, output_dim]

    # |Y - y|
    term1 = torch.mean(torch.abs(y_samples - y_true_exp), dim=(2, 3))  # [seq, batch]

    # |Y - Y'|
    diff = torch.abs(y_samples.unsqueeze(2) - y_samples.unsqueeze(3))  # [seq,batch,S,S,output_dim]
    term2 = 0.5 * torch.mean(diff, dim=(2,3,4))  # [seq, batch]

    crps = torch.mean(term1 - term2, dim=1)  # average over batch
    return crps  # shape [seq]

def coverage(y_true, y_samples, alpha=0.9):
    """
    y_true: [seq, batch, output_dim]
    y_samples: [seq, batch, n_samples, output_dim]
    returns: coverage_seq [seq]
    """
    lower = torch.quantile(y_samples, (1 - alpha) / 2, dim=2)
    upper = torch.quantile(y_samples, 1 - (1 - alpha) / 2, dim=2)
    covered = ((y_true >= lower) & (y_true <= upper)).float()
    coverage_seq = covered.mean(dim=(1, 2))  # average over batch and output_dim
    return coverage_seq

def crps_gaussian(y_true, mu, sigma):
    """
    CRPS for Gaussian predictive distributions computed per sequence step.
    Inputs:
      y_true: [seq, batch, output_dim]
      mu:     [seq, batch, output_dim]
      sigma:  [seq, batch, output_dim]  (sigma > 0)
    Returns:
      crps_seq: [seq]  (CRPS averaged over batch and output_dim)
    """
    # Ensure shapes match
    assert y_true.shape == mu.shape == sigma.shape, "shapes must match"

    # Numerical safety
    sigma = sigma.clamp_min(EPS)

    # Flatten last two dims to compute averages conveniently if desired
    # Compute z = (y - mu) / sigma
    z = (y_true - mu) / sigma  # shape [seq, batch, output_dim]

    # Standard normal cdf and pdf at z using Normal
    std_normal = Normal(loc=0.0, scale=1.0)
    cdf_z = std_normal.cdf(z)                 # Phi(z)
    pdf_z = torch.exp(std_normal.log_prob(z)) # phi(z)

    # CRPS formula for Gaussian:
    # crps = sigma * ( z*(2*Phi(z)-1) + 2*phi(z) - 1/sqrt(pi) )
    inv_sqrt_pi = 1.0 / math.sqrt(math.pi)
    crps_per_elem = sigma * ( z * (2.0 * cdf_z - 1.0) + 2.0 * pdf_z - inv_sqrt_pi )

    # average over batch and output_dim, keep seq dim
    seq_len = y_true.shape[0]
    # reshape to [seq, -1] then mean over axis 1
    crps_seq = crps_per_elem.reshape(seq_len, -1).mean(dim=1)
    return crps_seq  # [seq]


def coverage_gaussian(y_true, mu, sigma, alpha=0.9):
    """
    Empirical coverage for central (1-alpha) interval (e.g., alpha=0.9 => 90% interval)
    Inputs:
      y_true: [seq, batch, output_dim]
      mu:     [seq, batch, output_dim]
      sigma:  [seq, batch, output_dim]
      alpha:  float in (0,1) -> nominal coverage (default 0.9)
    Returns:
      coverage_seq: [seq]  fraction of y_true inside the central alpha-interval,
                          averaged over batch and output_dim.
    """
    assert 0.0 < alpha < 1.0

    sigma = sigma.clamp_min(EPS)

    # z quantiles for standard normal
    std_normal = Normal(0.0, 1.0)
    lower_p = (1.0 - alpha) / 2.0
    upper_p = 1.0 - lower_p
    # icdf accepts tensor input; create scalars on same device/dtype as mu
    device = mu.device
    dtype = mu.dtype
    z_lower = std_normal.icdf(torch.tensor(lower_p, device=device, dtype=dtype))
    z_upper = std_normal.icdf(torch.tensor(upper_p, device=device, dtype=dtype))

    # lower and upper bounds
    lower = mu + sigma * z_lower
    upper = mu + sigma * z_upper

    covered = ((y_true >= lower) & (y_true <= upper)).float()  # [seq, batch, output_dim]
    seq_len = y_true.shape[0]
    coverage_seq = covered.reshape(seq_len, -1).mean(dim=1)  # [seq]
    return coverage_seq
