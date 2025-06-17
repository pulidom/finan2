import sys; sys.path.insert(0, '../')
import numpy as np
from read_data import load_ts
import arbitrage as ar
import matplotlib.pyplot as plt
from matplotlib.dates import YearLocator
import cointegration as co

class cnf:
    pathdat='../dat/'
    tipo='asset' # 'asset', 'return', 'log_return', 'log'
    mtd = 'kf'# 'kf' 'exp' 'on' 'off'
    Ntraining = 1000 # length of the training period
    beta_win=61   #21
    zscore_win=31 #11
    sigma_co=1.5 # thresold to buy
    sigma_ve=0.1 # thresold to sell
    nmax=10#-1 # number of companies to generate the pairs (-1 all, 10 for testing)
    nsel=100# 100 # number of best pairs to select
    fname=f'../tmp/all_pair_{mtd}_' # fig filename
    #industry='oil'
    industry='beverages'
            
# load data
assets=['KO','PEP.O']
day,date,price,company = load_ts(assets=assets,sector=cnf.industry, pathdat=cnf.pathdat)
print(price.shape)
coca=price[5*252:,0]; pepsi=price[5*252:,1]
#coca=price[:,0]; pepsi=price[:,1]
date=date[5*252:]
dcoca=coca[1:]-coca[:-1]
dpepsi=pepsi[1:]-pepsi[:-1]
print(co.adf_test(coca))
print(co.adf_test(pepsi))
print(co.adf_test(dcoca))
print(co.adf_test(dpepsi))
spread1,_=co.calculate_spread_off(coca,pepsi)
spread2,_=co.calculate_spread_off(pepsi,coca)
print('co2pe',co.adf_test(spread1))
print('pe2co',co.adf_test(spread2))
zscore1,_,_ = co.off_zscore(spread1,cnf.zscore_win)
zscore2,_,_ = co.off_zscore(spread2,cnf.zscore_win)

figfile=cnf.fname+'asset1.png'
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(date,coca-coca.mean(),label='KO')
ax.plot(date,pepsi-pepsi.mean(),label='PEP.O')
ax.legend(frameon=False)
ax.tick_params(axis='x',rotation=60, zorder=120)
ax.xaxis.set_major_locator(YearLocator(1,month=1,day=1))
ax.set(ylabel='Price',xlabel='Year')
plt.tight_layout()
fig.savefig(figfile)
plt.close()
figfile=cnf.fname+'asset2.png'
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(date[1:],dcoca,label='KO')
ax.plot(date[1:],dpepsi,label='PEP.O')
ax.legend(frameon=False)
ax.tick_params(axis='x',rotation=60, zorder=120)
ax.xaxis.set_major_locator(YearLocator(1,month=1,day=1))
ax.set(ylabel='Price',xlabel='Year')
plt.tight_layout()
fig.savefig(figfile)
plt.close()

figfile=cnf.fname+'asset4.png'
fig, ax = plt.subplots(1,1,figsize=(6,4))
ax.plot(date,zscore1,label='KO')
#ax.plot(date,zscore2,label='PEP.O')
ax.legend(frameon=False)
ax.tick_params(axis='x',rotation=60, zorder=120)
ax.xaxis.set_major_locator(YearLocator(1,month=1,day=1))
ax.set(ylabel='Price',xlabel='Year')
plt.tight_layout()
fig.savefig(figfile)
plt.close()

figfile=cnf.fname+'asset5.png'
fig, ax = plt.subplots(1,1,figsize=(6,4))
#ax.plot(date,zscore1,label='KO')
ax.plot(date,zscore2,label='PEP.O')
ax.legend(frameon=False)
ax.tick_params(axis='x',rotation=60, zorder=120)
ax.xaxis.set_major_locator(YearLocator(1,month=1,day=1))
ax.set(ylabel='Price',xlabel='Year')
plt.tight_layout()
fig.savefig(figfile)
plt.close()
