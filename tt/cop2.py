from scipy.stats import norm, multivariate_normal

copula = GaussianCopula()
copula.fit(u, v)
z_scores = copula.conditional_zscore(u, v)
return 0, z_scores, copula

def gaussian_copula_log_likelihood(theta, u, v):
    rho = np.tanh(theta[0])  # Constrain rho to [-1, 1]
    cov = np.array([[1.0, rho], [rho, 1.0]])
    inv_cov = np.linalg.inv(cov)
    log_det = np.log(np.linalg.det(cov))
    
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    z = np.column_stack([z_u, z_v])  # Shape (n_samples, 2)
    
    # Compute z^T * inv_cov * z for each observation
    quad_form = np.sum(z.dot(inv_cov) * z, axis=1)  # Shape (n_samples,)
    
    # Sum over all observations
    nll = -0.5 * np.sum(quad_form + log_det)
    return -nll  # Minimize negative log-likelihood

def generate_signals(u, v, rho, threshold=0.95):
    z_u = norm.ppf(u)
    z_v = norm.ppf(v)
    cond_mean = rho * z_u
    cond_std = np.sqrt(1 - rho**2)
    prob_v_given_u = norm.cdf(z_v, loc=cond_mean, scale=cond_std)
    
    # Long when prob < 0.05, short when prob > 0.95
    signals = np.zeros(len(u))
    signals[prob_v_given_u < (1 - threshold)] = 1  # Y is too low relative to X
    signals[prob_v_given_u > threshold] = -1       # Y is too high relative to X

    #Transformo la condicional para luego calcular el z-score
    prob_cond = self.conditional_probability(u, v)
        
    # Transformar a Z-score: prob → z
    z_score = norm.ppf(prob_cond)

    return signals

# Rank-transform to uniform margins (PIT)
u = np.argsort(np.argsort(x)) / (len(x) + 1)
v = np.argsort(np.argsort(y)) / (len(y) + 1)

# Fit Gaussian copula
initial_theta = 0.5
res = minimize(gaussian_copula_log_likelihood, initial_theta, args=(u, v), method='BFGS')
rho = np.tanh(res.x[0])  # Estimated correlation
signals = generate_signals(u, v, rho, threshold=0.95)
