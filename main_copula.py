import numpy as np
import scipy.stats as ss
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
from copulas.multivariate import GaussianMultivariate
from copulas.bivariate import Gumbel
from copulas.bivariate import Frank
from copulas.bivariate import Clayton
from copulas.univariate import GaussianUnivariate
import copulas
from statsmodels.distributions.empirical_distribution import ECDF
import pandas as pd
#
from read_data import load_ts
import arbitrage as ar
import cointegration as co
import copula as cop

class cnf:
    pathdat='dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'kf'# 'kf' 'exp' 'on' 'off'
    Ntraining = 1000 # length of the training period
    beta_win=61   #21
    zscore_win=31 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=10#-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    fname=f'tmp/all_pair_{mtd}_' # fig filename
    #industry='oil'
    industry='beverages'
#['MGPI.O' 'SAM']
#['MGPI.O' 'PRMB.K']
#['MGPI.O' 'CCEP.O']
#['PEP.O' 'SAM']
#['MGPI.O' 'PEP.O']

assets=['KO','PEP.O']#['MGPI.O','SAM'] #['CIVI.K','NOG'] #['UEC','TNK']#['REX','WTI']#'MGPI.O','PRMB.K']
day,date,price,company,_ = load_ts(assets=assets,sector=cnf.industry, pathdat=cnf.pathdat)

def empirical_cdf(data):
    ranks = ss.rankdata(data)
    return ranks / (len(data) + 1)

def to_uniform_margins(data):
    ecdf = ECDF(data)
    return ecdf(data)

x=price[:,0]
y=price[:,1]

# Rank-transform to uniform marginals (PIT)
#u = np.argsort(np.argsort(price[:,0])) / (len(x) + 1)
#v = np.argsort(np.argsort(price[:,1])) / (len(y) + 1)

#u = to_uniform_margins(x)
#v = to_uniform_margins(y)

u = empirical_cdf(price[:,0])
v = empirical_cdf(price[:,1])
data_uv = np.column_stack((u, v))


from scipy.stats import t, multivariate_normal, multivariate_t, norm


class Gaussian:
    def __init__(self):
        self.rho = None  # correlación
        self.corr_matrix = None

    def fit(self, data):
        """
        data: np.ndarray of shape (n_samples, 2), assumed to be uniform marginals (in [0,1])
        """
        if data.shape[1] != 2:
            raise ValueError("Only bivariate Gaussian copula is supported.")

        # Transform to normal marginals
        norm_data = norm.ppf(data)

        # Estimate correlation matrix
        self.corr_matrix = np.corrcoef(norm_data.T)
        self.rho = self.corr_matrix[0, 1]

    def sample(self, n_samples):
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before sampling.")

        mean = np.zeros(2)
        samples = multivariate_normal.rvs(mean=mean, cov=self.corr_matrix, size=n_samples)

        # Transform back to uniform
        return norm.cdf(samples)

    def log_likelihood(self, data):
        """
        Computes the log-likelihood of the copula for uniform marginals.
        """
        if self.corr_matrix is None:
            raise ValueError("Model must be fit before computing log-likelihood.")

        norm_data = norm.ppf(data)
        mvn_logpdf = multivariate_normal.logpdf(norm_data, mean=np.zeros(2), cov=self.corr_matrix)
        ind_logpdf = np.sum(norm.logpdf(norm_data), axis=1)
        return mvn_logpdf - ind_logpdf

#class Clayton2(Clayton):
#    def __init__():
#
#    
#    def log_likelihood(self, data):
#        """
#        data: np.ndarray of shape (n_samples, 2) with uniform marginals
#        copula: fitted Clayton copula from copulas.bivariate
#        """
#        u, v = data[:, 0], data[:, 1]
#        theta = copula.theta
#
#        # Asegurarse de que no haya ceros que den problemas con log
#        eps = 1e-10
#        u = np.clip(u, eps, 1 - eps)
#        v = np.clip(v, eps, 1 - eps)
#
#        log_c = (
#            np.log(theta + 1)
#            - (theta + 1) * (np.log(u) + np.log(v))
#            - (2 + 1 / theta) * np.log(u ** (-theta) + v ** (-theta) - 1)
#        )
#
#        return log_c
#

class Student:
    def __init__(self):
        self.df = None
        self.rho = None
        self.cov = None

    def fit(self, data):
        # Transform uniform marginals to t quantiles (inverse CDF)
        u = data[:,0]
        v = data[:,1]
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        self.df = 5  # you can optimize this later
        x = t.ppf(u, df=self.df)
        y = t.ppf(v, df=self.df)

        # Estimate linear correlation
        rho = np.corrcoef(x, y)[0, 1]
        self.rho = rho
        self.cov = np.array([[1, rho], [rho, 1]])

    def sample(self, n):
        # Sample from bivariate t-distribution
        samples = multivariate_t.rvs(loc=[0, 0], shape=self.cov, df=self.df, size=n)
        # Transform back to uniform marginals
        u = t.cdf(samples[:, 0], df=self.df)
        v = t.cdf(samples[:, 1], df=self.df)
        return np.column_stack((u, v))

    def log_likelihood(self, data):
        u = data[:,0]
        v = data[:,1]
        u = np.clip(u, 1e-6, 1 - 1e-6)
        v = np.clip(v, 1e-6, 1 - 1e-6)
        x = t.ppf(u, df=self.df)
        y = t.ppf(v, df=self.df)
        z = np.column_stack((x, y))

        joint = multivariate_t.pdf(z, loc=[0, 0], shape=self.cov, df=self.df)
        marg_x = t.pdf(x, df=self.df)
        marg_y = t.pdf(y, df=self.df)
        copula_density = joint / (marg_x * marg_y)
        return np.log(copula_density)

    
# Lista de clases
copula_classes = [Gaussian,
                  Student,
                  Clayton,
                  Gumbel,
                  Frank]

# Instanciar y almacenar en un diccionario
Copulas = {cls.__name__: cls() for cls in copula_classes}

samples=[]
for name, Copula in Copulas.items():
    Copula.fit(data_uv)
    sample = Copula.sample(1000)
    print( sample.shape )
    samples.append(sample)
#
#
#results = []; samples=[]
#for name, Copula in Copulas.items():
#    print(name,u.shape,v.shape)
#    Copula.fit(data_uv)
#    log_likelihood = Copula.log_likelihood(data_uv)
#    n_params = 1  # Para la mayoría (Gaussian, Clayton, Gumbel, Frank)
#    if name == "Student-t":
#        n_params = 2  # rho y df
#    aic = -2 * log_likelihood + 2 * n_params
#    bic = -2 * log_likelihood + n_params * np.log(len(u))
#    results.append({
#        "Copula": name,
#        "Log-Likelihood": log_likelihood,
#        "AIC": aic,
#        "BIC": bic,
#        "Params": f"θ={getattr(Copula, 'theta', getattr(Copula, 'rho', None)):.3f}" + 
#                 (f", df={Copula.df:.1f}" if hasattr(Copula, 'df') else "")
#    })
#    
#    samples.append(Copula.sample(1000))
## Resultados en una tabla
#results_df = pd.DataFrame(results)
#print(results_df.sort_values(by="AIC"))
#
#quit()

copula2 = GaussianMultivariate()
copula2.fit(data_uv)

n=1000
# 5. Sample from the copula (optional)
samples2 = copula2.sample(n).to_numpy()

#sampling = {
#    "Gaussian": cop.GaussianCopula().sample(n),
#    "Student-t": cop.StudentTCopula().sample(n),
#    "Clayton": cop.ClaytonCopula().sample(n),
#    "Gumbel": cop.GumbelCopula().sample(n),
#    "Frank": cop.FrankCopula().sample(n),
#}

# 6. Plot original vs copula-generated

#fig, axs = plt.subplots(1, 2, figsize=(12, 5))
#sns.kdeplot(x=u, y=v, fill=True, ax=axs[0])
#axs[0].set_title('transformed CDF (u,v)')
#
#sns.kdeplot(x=samples[:,0], y=samples[:,1], fill=True, ax=axs[1])
#axs[1].set_title('Samples from Gaussian Copula Model')
#plt.tight_layout()
#figfile=cnf.fname+'distributions.png'
#fig.savefig(figfile)
##plt.show()
#

i=0
for name, Copula in Copulas.items():
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    sns.kdeplot(x=u, y=v, fill=True, ax=axs[0])
    axs[0].set_title('Original Transformed Data (u,v)')

    sns.kdeplot(x=samples[i][:,0], y=samples[i][:,1], fill=True, ax=axs[1])
    axs[1].set_title('Samples from Copula Model'+name)

    plt.tight_layout()
    figfile=cnf.fname+f'distributions-{name}.png'
    fig.savefig(figfile)

    i+=1
    
#plt.show()


quit()

nt=price.shape[1]
jlasts=range(cnf.Ntraining+cnf.Njump,nt,cnf.Njump)

res_tt=[]
iini=0
for jlast in jlasts:
    assets_tr=price[:,iini:jlasts]
    iini+=cnf.Njump
    res = ar.given_pairs(price,assets,cnf)
    res_tt.append( res )



figfile=cnf.fname+'asset'+assets[0]+'-'+assets[1]+'.png'
print(figfile)
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(date,price[:,0],label=assets[0])
ax.plot(date,price[:,1],label=assets[1])
ax.legend(frameon=False)
ax.tick_params(axis='x',rotation=60, zorder=120)
ax.xaxis.set_major_locator(YearLocator(1,month=1,day=1))
ax.set(ylabel='Price',xlabel='Year')
plt.tight_layout()
plt.show()
fig.savefig(figfile)
plt.close()


figfile=cnf.fname+'capital_'+assets[0]+'-'+assets[1]+'.png'
fig, ax = plt.subplots(1,1,figsize=(8,4))
for res in res_tt:
    ax.plot(res.capital[:20,:].mean(0))
plt.tight_layout()
fig.savefig(figfile)
plt.close()
