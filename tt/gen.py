def generate_data2(n=1500):
    """Genera datos mas sencillo univariado"""
    z = np.random.uniform(0., 1, size=(n, 1))
    mu_z =  np.sin(2 * np.pi * z)
    sigma_z = 0.2 * (1- 0.6*np.cos(2*np.pi*z)) #0.2 * np.sqrt((1 - 2 * z[:, 0]) * (1 - 2 * z[:, 1]))
    x = np.random.normal(mu_z, sigma_z)
    return x.squeeze(), z
