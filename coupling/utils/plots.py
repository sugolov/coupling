import numpy as np
from torch.autograd import grad

import torch
import matplotlib.pyplot as plt

import time
import gc
from functorch.experimental import chunk_vmap

def plotsvals(J, h, fh, title=None, device='cuda', SVD=SVD, label='0'):
    import matplotlib.colors as mcolors
    pcolors = sorted(list(mcolors.TABLEAU_COLORS.values()))
    svr = 1
    U, S, V = SVD(J)
    #U, S, V = svd(J) # non random svd
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in J]
    U, V = [u[..., :K+E] for u in U], [v[..., :K+E] for v in V]
    #U, V = [u[:, :K+E] for u in U], [v[ :K+E, :].T for v in V] #non random
    print(S.shape)

    d = torch.arange(len(S)).float() + 1
    csfont = {'fontname':'Times New Roman'}
    fs = 40
    fig, axs = plt.subplots(1, 2, figsize=(30,12))
    for i in range(15):
        if svr:
            if i == 0:
                #axs[0].scatter(d, 1 / S[:, i])
                axs[0].scatter(d, 1 / (S[:, i]))
            #axs[1].scatter(d, S[:, i], color=pcolors[9 * int(i<10)])
            axs[1].scatter(d, S[:, i], color=pcolors[9 * int(i<10)])
        else:
            axs[0].scatter(d, S[:, i])
    if svr:
        axs[0].tick_params(labelsize=40)
        axs[1].tick_params(labelsize=40)
        axs[0].set_xlabel('Depth', fontsize=fs, **csfont)
        axs[0].set_ylabel('1 / Singular Value', fontsize=fs, **csfont)
        axs[1].set_xlabel('Depth', fontsize=fs, **csfont)
        axs[1].set_ylabel('Singular Value', fontsize=fs, **csfont)
        # axs[0].legend(np.arange(K+E)+1, fontsize=40)
        axs[1].legend(np.arange(K+E)+1, fontsize=17)
    else:
        axs[0].set_xlabel('Depth', fontsize=fs, **csfont)
        axs[0].set_ylabel('Singular Value', fontsize=fs, **csfont)
        axs[0].legend(np.arange(K+E)+1, fontsize=15)

    def fit_line(x, y, ax, plot=False):
        xm = torch.mean(x)
        ym = torch.mean(y)
        x_ = x - xm
        a = torch.sum(x_*y)/torch.sum(x_**2)
        b = ym - a*xm
        y_hat = a*x + b
        if plot:
            ax.plot(x, y_hat)
        rss = torch.sum((y - y_hat)**2)
        tss = torch.sum((y - ym)**2)
        rsq = 1 - rss/tss
        return rsq.item()

    for i in range(1):
        if svr:
            print('fit_line', fit_line(d[9:], 1 / S[9:,i], axs[0], True))
        else:
            print('fit_line', fit_line(d, S[:,i], axs[0], True))

    # plt.title(title)
    plt.savefig(title + 'svals{}.png'.format(label), bbox_inches = 'tight')
    plt.close()

    if svr:
        return J, U, S, V
    else:
        return None

def alignment(h, fh, title=None, device='cuda', JUSV=None, SVD=SVD, label='0'):
    if JUSV is None:
        J = [jacobian(fh[l], h[l-1], device).cpu() for l in range(1, len(h))]
        U, S, V = SVD(J)
        S = torch.stack(S).cpu()
    else:
        J, U, S, V = JUSV
    J = [j.cpu() for j in J]
    U = [u.cpu() for u in U]
    V = [v.cpu() for v in V]
    V = V

    fig, axs = plt.subplots(len(S), len(S), figsize=(20,20))
    for i in range(len(S)):
        for j in range(len(S)):
            ax = axs[i,j]
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            uj, ji, vj = U[j], J[i], V[j]
            if uj.shape[0] != ji.shape[0] or vj.shape[0] != ji.shape[1]:
              print("wrong shape")
              continue
            m = uj[:,:K].T @ ji @ vj[:,:K]
            a = ax.matshow(torch.abs(m).detach().cpu().numpy(), cmap='RdBu')
    plt.savefig(title + 'UJV{}.png'.format(label), bbox_inches = 'tight')
    plt.close()

    fig, axs = plt.subplots(len(S), len(S), figsize=(20,20))
    for i in range(len(S)):
        for j in range(len(S)):
            ax = axs[i,j]
            ax.tick_params(left=False, right=False, labelleft=False, labelbottom=False, bottom=False)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            uj, ji, vj = U[j], J[i], V[j]
            if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
                continue
            m = vj[:,:K].T @ ji @ uj[:,:K]
            a = ax.matshow(torch.abs(m).detach().cpu().numpy(), cmap='RdBu')
    plt.savefig(title + 'VJU{}.png'.format(label), bbox_inches = 'tight')
    plt.close()

def plots(J, h, fh, title=None, device='cuda'):
    JUSV = plotsvals(J, h, fh, title, device, randomsvd, '1')
    alignment(h, fh, title, device, JUSV, randomsvd, '1')