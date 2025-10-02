
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
import read_data as rd
import stats as mystats
from PyIF.te_compute import te_compute

def meanvar_ts(day,price,ndegree=5):
    ' estimate mean and variance of n time series '
    coeffs =  np.polyfit(day, price, ndegree)
    print(coeffs.shape)
    mean_pred = np.array([np.polyval(coeffs[:,i], day) for i in range(price.shape[1])]).T
    pert = price - mean_pred
    return mean_pred, pert.var(0)

def copula_transform(x):
    """Gaussian copula transformation to normalize data."""
    ranks = stats.rankdata(x) / (len(x) + 1) 
    return stats.norm.ppf(ranks)  # Transform to standard normal (Gaussian copula)

def mi(x,y):
    ''' Determines mutual information between two time series applying copula first 
           (better to remove mean before calling this function) '''
    
    x_copula = copula_transform(x)
    y_copula = copula_transform(y)

    mi = mutual_info_regression(x_copula.reshape(-1, 1), y_copula, discrete_features=False)[0]
    return mi

def kld(sample, nbin=50):
    ''' Measure KLD of the sample distribution from the gaussian
	Nongaussianity measure
    '''
    mu=sample.mean()
    sigma=sample.std()
	
    #deberia usar kde o el histograma para calcular la distribucion?
    hist,bins =np.histogram(sample,bins=nbin, density=True)
    dbin=bins[1]-bins[0]
    cbin=0.5*(bins[1:]+bins[:-1]) # centers of the bins
    gaussian_pdf = stats.norm.pdf(cbin, mu, sigma)
	
    # Avoid division by zero and ensure numerical stability
    epsilon = 1e-10
    gaussian_pdf = np.clip(gaussian_pdf, epsilon, None)
    hist = np.clip(hist, epsilon, None)

    # Compute KL divergence
    kld = np.sum(hist * np.log(hist / gaussian_pdf)) * dbin	
    #Hsample = -(hist*np.log(np.abs(hist))).sum()	
    #Hgauss=0.5 np.log (2 * np.e * np.pi * sigma**2 )
    return kld

def order_company(price,company,ncomp=10):
    mean,var = mystats.meanvar_ts(day,price)
    price_per = price - mean
    ntot=len(company)
    x=price_per[0]
    
    corr,mi1,mi2,te = np.zeros((4,ntot-1))
    
    for i in range(ntot-1):
        corr[i]=np.corrcoef(x,price_per[i+1])[0,1]
        mi1[i]=mutual_info_regression(x[:,None], price_per[i+1,:], discrete_features=False)[0]
        print(i,company[i+1],mi1[i])
        mi2[i]= mi(x, price_per[i+1])
        te[i] = te_compute(x,price_per[i+1])

    price_rest = price_per[1:] # saco la autoref
    company_rest = company[1:]
    idx = np.argsort(corr)
    corr_ordered= corr[idx][::-1][:ncomp]
    company_ordered1=company_rest[idx][::-1][:ncomp]
    price1 =  np.concatenate((x[None,:], price_rest[idx][::-1][:ncomp]))
    idx = np.argsort(mi1)
    mi1_ordered= mi1[idx][::-1][:ncomp]
    price2 =  np.concatenate((x[None,:], price_rest[idx][::-1][:ncomp]))
    company_ordered2=company_rest[idx][::-1][:ncomp]
    idx = np.argsort(mi2)
    mi2_ordered= mi2[idx][::-1][:ncomp]
    company_ordered3=company_rest[idx][::-1][:ncomp]
    price3 =  np.concatenate((x[None,:], price_rest[idx][::-1][:ncomp]))
    idx = np.argsort(te)
    te_ordered= te[idx][::-1][:ncomp]
    company_ordered4=company_rest[idx][::-1][:ncomp]
    price4 =  np.concatenate((x[None,:], price_rest[idx][::-1][:ncomp]))

    return (corr_ordered, company_ordered1, price1,
            mi1_ordered,  company_ordered2, price2,
            mi2_ordered,  company_ordered3, price3,
            te_ordered,   company_ordered4, price4,
            )

def lag_mi(price,nt=30):
    x=price[0]
    lag_mi=np.zeros((price.shape[0]-1,nt))
    for icompany in range(price.shape[0]-1):
        lag_mi[icompany,0] = mutual_info_regression(
            x[:,None], price[icompany+1,:], discrete_features=False)[0]
        print(icompany,lag_mi[icompany,0])
        for it in range(1,nt):
            lag_mi[icompany,it] = mutual_info_regression(
                x[it:,None], price[icompany+1,:-it], discrete_features=False)[0]
            # cuanto predigo usando el pasado de la otra serie
    return lag_mi

'''def transfer_entropy(price,nt=30):
    x=price[0]
    lag_mi=np.zeros((price.shape[0]-1,nt))
    for icompany in range(price.shape[0]-1):
        lag_mi[icompany,0] = te_compute(
            x[:,None], price[icompany+1,:])
        print(icompany,lag_mi[icompany,0])
        for it in range(1,nt):
            lag_mi[icompany,it] = mutual_info_regression(
                x[it:,None], price[icompany+1,:-it], discrete_features=False)[0]
            # cuanto predigo usando el pasado de la otra serie
    return lag_mi
'''
def plot_histogram(price,company):
    ' grafica 6 paneles con los histogramas y la gaussiana de los datos '
    # https://stackoverflow.com/questions/20011122/fitting-a-normal-distribution-to-1d-data/20012350#20012350
    for j in range(2):
        figfile=f'tmp/histo_{j+1}.png'
        fig, ax = plt.subplots(1,3,figsize=(9,3))
        for i in range(3):
            ax[i].hist(price[i+j*3],bins=30,rwidth=0.8, alpha=0.6, density=True)
            mu,std=price[i+j*3].mean(),price[i+j*3].std()
            xmin, xmax = ax[i].get_xlim()    
            x = np.linspace(xmin, xmax, 100)
            p = stats.norm.pdf(x, mu, std)
            ax[i].plot(x, p, 'k', linewidth=2)
            ax[i].set_title(company[i+j*3])
        plt.tight_layout()
        fig.savefig(figfile)
        plt.close()
    
            
if __name__=="__main__":

    day,price,company = rd.load_n_ts(jref=-1,nvar=20,)
    print(company)
    mean,var = mystats.meanvar_ts(day,price)
    price_per = price - mean
    x=price_per[0]
    for p in price_per:
        print('nongauss:',kld(p))
    m1,c1,p1,m2,c2,p2,m3,c3,p3,m4,c4,p4 = order_company(price_per,company,ncomp=5)
    print('correlation')
    print(c1)
    print(m1)
    print('mi')
    print(c2)
    print(m2)
    print('mi + copula')
    print(c3)
    print(m3)
    print('Te')
    print(c4)
    print(m4)

    milag = lag_mi(p2)
    print('Transfer entropy')
    for i in range(1,6):
        print(te_compute(p2[i],p2[0]))
    quit()
    
    figfile=f'tmp/mi_1.png'
    fig, ax = plt.subplots(1,1,figsize=(4,3))
    print(milag.shape)
    for j,mi in enumerate(milag):
        print(mi.shape)
        ax.plot(mi,label=c2[j])
    ax.set(xlabel='Time lag',ylabel='MI')
    ax.legend()
    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()
    plot_histogram(price_per,company)
    quit()
    for i in range(20):
        corr = np.corrcoef(x,price_per[i+1])[0,1]
        mi1 = mutual_info_regression(x[:,None], price_per[i+1,:], discrete_features=False)
        mi2 = mi(x, price[i+1])
        print(mi1,mi2,corr)
        print('nongauss:',kld(price_per[i+1]))
    print('Completo sin la perturba')
    x=price[0]
    for i in range(20):
        corr = np.corrcoef(x,price[i+1])[0,1]
        mi1 = mutual_info_regression(x[:,None], price[i+1,:], discrete_features=False)
        mi2 = mi(x, price[i+1])
        print(mi1,mi2,corr)
        
