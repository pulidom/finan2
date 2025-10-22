import numpy as np, os, copy
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import utils

def vertical_bar(axs,compras,ccompras):
    ''' Plot the entrada y salida de posiciones '''
    start_indices, end_indices= utils.calc_startend(compras[:,None])
    start_cindices, end_cindices= utils.calc_startend(ccompras[:,None])

    indices=np.arange(compras.shape[0])
    
    t=np.arange(compras.shape[0])/252
    for ax in axs:
        for start, end in zip(start_indices[0], end_indices[0]):
            ax.axvspan(t[start], t[end], alpha=0.3, color='green')
        for start, end in zip(start_cindices[0], end_cindices[0]):
            ax.axvspan(t[start], t[end], alpha=0.3, color='red')
    
def plot_zscore(j,res0,fname):

    res = copy.deepcopy(res0)
    print(res0.spread.shape)
    
    if res0.spread.ndim==2:
        nt=res0.spread.shape[1]# 1-->0???
        res.reorder(j) # select the pair
    else:
        nt=res0.spread.shape[0]# 1-->0???

    t=np.arange(nt)/252
    
    figfile=fname+f'zscore{j}.png'
    fig = plt.figure(figsize=(7, 5))

    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1])

    ax1 = fig.add_subplot(gs[0, 0])
#    ax1.plot(res.assets[0],label=res.company[0])
#    ax1.plot(res.assets[1],label=res.company[1])
    ax1.plot(t,res.asset_x,label='x')#,label=res.company[0])
    ax1.plot(t,res.asset_y,label='y')#,label=res.company[1])
    ax1.legend()
    ax1.set_title('Assets')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(t,res.spread)
    ax2.plot(t,res.spread_mean)
    ax2.fill_between(t, res.spread_mean - 1.96* res.spread_std,
                     res.spread_mean +1.95* res.spread_std,color='gray', alpha=0.2)
    ax2.set_title('Spread')

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(t,res.zscore)
    vertical_bar([ax3],res.compras,res.ccompras)
    #vertical_bar([ax3],res.compras,res.ccompras)
    ax3.set_title('Z-score')
    
    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()

    
def plot_capital_single(j,res0,fname):
    nt=res0.spread.shape[1]
    res = copy.deepcopy(res0) 
    res.reorder(j) # select the pair
    
    figfile=fname+f'capital{j}.png'
    
    fig, ax = plt.subplots(3,1,figsize=(7,7))
    ax[0].plot(res.zscore)
    ax[0].set_title('Z-score')
    

    for ivar in range(res.corto.shape[-1]):
        ax[1].plot(res.corto[:,ivar],label='corto '+res.company[ivar])
    ax[1].legend()

    for ivar in range(res.corto.shape[-1]):
        ax[2].plot(res.largo[:,ivar],label='largo '+res.company[ivar])
    ax[2].legend()

    vertical_bar(ax,res.compras,res.ccompras)

    plt.tight_layout()
    fig.savefig(figfile)
    plt.close()
