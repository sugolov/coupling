import numpy as np
from torch.autograd import grad

import torch
import matplotlib.pyplot as plt

import time
import gc
from functorch.experimental import chunk_vmap

def randomevd(mm, d, k, l, its=0, device='cuda'):
    '''
    Approximates the top eigenvalues(vectors) of a matrix M.

    === Input parameters ==
    mm:
        function that takes in a matrix Q and computes MQ
    d:
        width of matrix M
    k:
        number of principal components to extract
    l:
        number of random vectors to project. Usually k + 3
    its:
        number of power iterations. Usually even 0 is good
    device:
        cpu or cuda

    === Output variables ===
    (Ds, Vs)
    Ds:
        Top eigenvalues of M, 1-D tensor of (k,)
    Vs:
        Top eigenvectors of M, 2-D tensor of (d, k)
    '''

    if l < k:
        raise ValueError("l={} less than k={}".format(l, k))
    Q = torch.randn(d, l).to(device)
    Q = mm(Q)
    Q, _ = torch.qr(Q)
    for i in range(its):
        Q = mm(Q)
        Q, _ = torch.qr(Q)
    R = Q.T @ mm(Q)
    R = (R + R.T) / 2
    D, S = torch.linalg.eigh(R, UPLO='U')
    V = Q @ S
    D, V = D[-k:], V[:, -k:]
    return D.flip(-1), V.flip(-1)

# K = 10
# L = 5
# E = 5
# ITS = 1
# K = 30
K = 30
#K = 40
L = 20
E = 5
ITS = 20


def randomsvd_each(J, K, L, E, ITS):
    dl = J.shape[0]
    dr = J.shape[1]
    k = K + E
    l = K + E + L
    its = ITS

    def mml(Q):
        #print(J.dtype)
        #print(Q.dtype)
        tmp = J.T @ Q
        return J @ tmp
    def mmr(Q):
        tmp = J @ Q
        return J.T @ tmp
    Dl, U = randomevd(mml, dl, k, l, its, 'cpu')
    Dr, V = randomevd(mmr, dr, k, l, its, 'cpu')
    #print(Dl - Dr)
    #print(Dl)
    return U, Dl ** 0.5, V

def randomsvd(J, K, L, E, ITS):
    Us, Ss, Vs = [], [], []
    for j in range(len(J)):
        # i = len(J) - j - 1
        i = j
        tt = time.time()
        print('svding layer {}'.format(i))
        U, S, V = randomsvd_each(J[i], K, L, E, ITS)
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
        print(time.time() - tt)
    return Us, Ss, Vs

def svd(J):
    Us, Ss, Vs = [], [], []
    for j in range(len(J)):
        # i = len(J) - j - 1
        i = j
        tt = time.time()
        print('svding layer {}'.format(i))
        U, S, V = torch.linalg.svd(J[i])
        Us.append(U)
        Ss.append(S)
        Vs.append(V)
        print(time.time() - tt)
    return Us, Ss, Vs

SVD = randomsvd

def jacobian(output, input, index, device):
    print(output.shape)
    tt = time.time()
    output = output[0, index, :]
    J = []
    for i in range(output.numel()):
      I = torch.zeros_like(output)
      ind = np.unravel_index(i, I.shape)
      I[ind] = 1
      j = grad(output, input, I, retain_graph=True)[0][0, index, :]
      #j = grad(output, input, I)[0][0, index, :]
      I[ind] = 0
      J.append(j.flatten())
    print(time.time()-tt)

    return torch.stack(J, dim=0)

    

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

def alignment_metric(Jac):
    #K = 30
    #K = 40
    diag_mat_all_k = {}
    alt_mat_all_k = {}
    alt_mat_sing_all_k = {}
    aln_uu_all_k = {}
    aln_vv_all_k = {}
    aln_uv_all_k = {}
    aln_vu_all_k = {}

    L = 20
    E = 5
    ITS = 20
    num_sing_vecs = [5, 10, 15, 20]
    #U_all, S, V_all = SVD(Jac, 300, L, E, ITS)
    U_all, S, V_all = svd(Jac) # nonrandom
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in Jac]
    
    for K in num_sing_vecs:
        U, V = [u[:, :K+E] for u in U_all], [v[ :K+E, :].T for v in V_all] #nonrandom
        #U, V = [u[..., :K+E] for u in U_all], [v[..., :K+E] for v in V_all]

        diag_mat = torch.zeros((len(S), len(S)))
        vju_diag_mat = torch.zeros((len(S), len(S)))

        alt_mat = torch.zeros((len(S), len(S)))
        alt_mat_col = torch.zeros((len(S), len(S)))
        alt_mat_row = torch.zeros((len(S), len(S)))

        alt_mat_normalized = torch.zeros((len(S), len(S)))
        alt_mat_normalized_sing = torch.zeros((len(S), len(S)))
        alt_mat_normalized_sing_by_sing = torch.zeros((len(S), len(S)))
        alt_mat_normalized_sing_by_sing_diag = torch.zeros((len(S), len(S)))
        
        vju_alt_mat = torch.zeros((len(S), len(S)))
        vju_alt_mat_col = torch.zeros((len(S), len(S)))
        vju_alt_mat_row = torch.zeros((len(S), len(S)))

        alt_mat_sing = torch.zeros((len(S), len(S)))
        alt_mat_sing_col = torch.zeros((len(S), len(S)))
        alt_mat_sing_row = torch.zeros((len(S), len(S)))

        vju_alt_mat_sing = torch.zeros((len(S), len(S)))
        vju_alt_mat_sing_col = torch.zeros((len(S), len(S)))
        vju_alt_mat_sing_row = torch.zeros((len(S), len(S)))

        aln_uu = torch.zeros((len(S), len(S)))
        aln_vv = torch.zeros((len(S), len(S)))
        aln_uv = torch.zeros((len(S), len(S)))
        aln_vu = torch.zeros((len(S), len(S)))

        aln_uu_row = torch.zeros((len(S), len(S)))
        aln_vv_row = torch.zeros((len(S), len(S)))
        aln_uv_row = torch.zeros((len(S), len(S)))
        aln_vu_row = torch.zeros((len(S), len(S)))

        aln_uu_col = torch.zeros((len(S), len(S)))
        aln_vv_col = torch.zeros((len(S), len(S)))
        aln_uv_col = torch.zeros((len(S), len(S)))
        aln_vu_col = torch.zeros((len(S), len(S)))

        print(S[0].shape)


        for i in range(len(S)):
            for j in range(len(S)):
                uj, ji, vj = U[j], J[i], V[j]
                ui, vi = U[i], V[i]
                if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
                    print("wrong shape")
                    continue
                m1 = vj[:,:K].T @ ji @ uj[:,:K]
                m2 = uj[:,:K].T @ ji @ vj[:,:K]

                m_uu = uj[:,:K].T @ ui[:,:K]
                m_vv = vj[:,:K].T @ vi[:,:K]
                m_uv = uj[:,:K].T @ vi[:,:K]
                m_vu = vj[:,:K].T @ ui[:,:K]

                Id = torch.eye(m_uu.shape[0])

                d1 = torch.diag(torch.diag(m1))
                
                s2 = torch.diag(S[i][:K])
                d2 = torch.diag(torch.diag(m2))
                
                diag_mat[i, j] = (torch.sum(torch.abs(m2-d2))/(K * (K-1))) / (torch.sum(torch.abs((d2)))/K)
                
                vju_diag_mat[i, j] = (torch.sum(torch.abs(m1-d1))/(K * (K-1))) / (torch.sum(torch.abs((d1)))/K)

                alt_mat[i, j] = torch.linalg.norm(m2-d2).mean()
                norm_m2 = torch.linalg.norm(m2)
                alt_mat_normalized[i, j] = torch.linalg.norm(m2-d2) / norm_m2

                alt_mat_row[i, j] = torch.linalg.norm(m2-d2, dim=1).mean()
                alt_mat_col[i, j] = torch.linalg.norm(m2-d2, dim=0).mean()

                vju_alt_mat[i, j] = torch.linalg.norm(m1-d1).mean()
                vju_alt_mat_row[i, j] = torch.linalg.norm(m1-d1, dim=1).mean()
                vju_alt_mat_col[i, j] = torch.linalg.norm(m1-d1, dim=0).mean()

                alt_mat_sing[i, j] = torch.linalg.norm(torch.abs(m2)-s2)
                alt_mat_sing_row[i, j] = torch.linalg.norm(torch.abs(m2)-s2, dim=1).mean()
                alt_mat_sing_col[i, j] = torch.linalg.norm(torch.abs(m2)-s2, dim=0).mean()

                norm_s2 = torch.linalg.norm(s2)

                alt_mat_normalized_sing[i, j] = torch.linalg.norm(torch.abs(m2)-s2) / norm_m2
                alt_mat_normalized_sing_by_sing[i, j] = torch.linalg.norm(torch.abs(m2)-s2) / norm_s2
                
                alt_mat_normalized_sing_by_sing_diag[i, j] = torch.linalg.norm(torch.abs(d2)-s2) / norm_s2

                vju_alt_mat_sing[i, j] = torch.linalg.norm(torch.abs(m1)-s2)
                vju_alt_mat_sing_row[i, j] = torch.linalg.norm(torch.abs(m1)-s2, dim=1).mean()
                vju_alt_mat_sing_col[i, j] = torch.linalg.norm(torch.abs(m1)-s2, dim=0).mean()

                uu = torch.abs(m_uu)-Id
                vv = torch.abs(m_vv)-Id
                uv = torch.abs(m_uv)-Id
                vu = torch.abs(m_vu)-Id

                aln_uu[i, j] = torch.linalg.norm(uu).mean()
                aln_vv[i, j] = torch.linalg.norm(vv).mean()
                aln_uv[i, j] = torch.linalg.norm(uv).mean()
                aln_vu[i, j] = torch.linalg.norm(vu).mean()

                aln_uu_row[i, j] = torch.linalg.norm(uu, dim=1).mean()
                aln_vv_row[i, j] = torch.linalg.norm(vv, dim=1).mean()
                aln_uv_row[i, j] = torch.linalg.norm(uv, dim=1).mean()
                aln_vu_row[i, j] = torch.linalg.norm(vu, dim=1).mean()

                aln_uu_col[i, j] = torch.linalg.norm(uu, dim=0).mean()
                aln_vv_col[i, j] = torch.linalg.norm(vv, dim=0).mean()
                aln_uv_col[i, j] = torch.linalg.norm(uv, dim=0).mean()
                aln_vu_col[i, j] = torch.linalg.norm(vu, dim=0).mean()

        diag_mat_all = {}
        diag_mat_all['ujv'] = diag_mat
        diag_mat_all['vju'] = vju_diag_mat

        diag_mat_all_k[K] = diag_mat_all


        alt_mat_all = {}
        alt_mat_all['frob'] = alt_mat
        alt_mat_all['row'] = alt_mat_row
        alt_mat_all['col'] = alt_mat_col
        alt_mat_all['frob_vju'] = vju_alt_mat
        alt_mat_all['row_vju'] = vju_alt_mat_row
        alt_mat_all['col_vju'] = vju_alt_mat_col

        alt_mat_all['frob_normalized'] = alt_mat_normalized

        alt_mat_all_k[K] = alt_mat_all


        alt_mat_sing_all = {}
        alt_mat_sing_all['frob'] = alt_mat_sing
        alt_mat_sing_all['row'] = alt_mat_sing_row
        alt_mat_sing_all['col'] = alt_mat_sing_col
        alt_mat_sing_all['frob_vju'] = vju_alt_mat_sing
        alt_mat_sing_all['row_vju'] = vju_alt_mat_sing_row
        alt_mat_sing_all['col_vju'] = vju_alt_mat_sing_col

        alt_mat_sing_all['frob_normalized'] = alt_mat_normalized_sing
        alt_mat_sing_all['frob_normalized_by_sing'] = alt_mat_normalized_sing_by_sing
        alt_mat_sing_all['frob_normalized_by_sing_diag'] = alt_mat_normalized_sing_by_sing_diag

        alt_mat_sing_all_k[K] = alt_mat_sing_all
        

        aln_uu_all = {}
        aln_uu_all['frob'] = aln_uu
        aln_uu_all['row'] = aln_uu_row
        aln_uu_all['col'] = aln_uu_col

        aln_uu_all_k[K] = aln_uu_all


        aln_vv_all = {}
        aln_vv_all['frob'] = aln_vv
        aln_vv_all['row'] = aln_vv_row
        aln_vv_all['col'] = aln_vv_col

        aln_vv_all_k[K] = aln_vv_all

        aln_uv_all = {}
        aln_uv_all['frob'] = aln_uv
        aln_uv_all['row'] = aln_uv_row
        aln_uv_all['col'] = aln_uv_col

        aln_uv_all_k[K] = aln_uv_all

        aln_vu_all = {}
        aln_vu_all['frob'] = aln_vu
        aln_vu_all['row'] = aln_vu_row
        aln_vu_all['col'] = aln_vu_col

        aln_vu_all_k[K] = aln_vu_all


    return diag_mat_all_k, alt_mat_all_k, alt_mat_sing_all_k, aln_uu_all_k, aln_vv_all_k, aln_uv_all_k, aln_vu_all_k
    ##alt_mat = torch.zeros((len(S), len(S)))
    #
    #aln_uu = torch.zeros((len(S), len(S)))
    #aln_vv = torch.zeros((len(S), len(S)))
    #aln_uv = torch.zeros((len(S), len(S)))
    #aln_vu = torch.zeros((len(S), len(S)))


    #for i in range(len(S)):
    #  for j in range(len(S)):
    #    uj, ji, vj = U[j], J[i], V[j]
    #    ui, vi = U[i], V[i]
    #    if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
    #      print("wrong shape")
    #      continue
    #    m2 = uj[:,:K].T @ ji @ vj[:,:K]
    #    
    #    m_uu = uj[:,:K].T @ ui[:,:K]
    #    m_vv = vj[:,:K].T @ vi[:,:K]
    #    m_uv = uj[:,:K].T @ vi[:,:K]
    #    m_vu = vj[:,:K].T @ ui[:,:K]

    #    Id = torch.eye(m_uu.shape[0])

    #    d2 = torch.diag(torch.diag(m2))
    #    diag_mat[i, j] = (torch.sum(torch.abs(m2-d2))/(K * (K-1))) / (torch.sum(torch.abs((d2)))/K)
    #    alt_mat[i, j] = torch.linalg.norm(m2-d2)
    #    
    #    aln_uu[i, j] = torch.linalg.norm(torch.abs(m_uu)-Id)
    #    aln_vv[i, j] = torch.linalg.norm(torch.abs(m_vv)-Id)
    #    aln_uv[i, j] = torch.linalg.norm(torch.abs(m_uv)-Id)
    #    aln_vu[i, j] = torch.linalg.norm(torch.abs(m_vu)-Id)
    #    


    #return diag_mat, alt_mat, aln_uu, aln_vv, aln_uv, aln_vu

def projection_similarity(U_A, U_B, normalize=True, device='cuda'):
    """
    Compute the projection similarity (trace overlap) between two subspaces.
    Related to the chordal (projection) metric on Grassmannian
    Arguments:
        U_A: Tensor of shape (n, k), orthonormal basis of subspace A.
        U_B: Tensor of shape (n, k), orthonormal basis of subspace B.
        normalize: Whether to normalize the projection similarity by k.
    Returns:
        projection_similarity: Trace of the projection overlap.
    """
    U_A = U_A.to(device)
    U_B = U_B.to(device)
    
    M = U_A.T @ U_B  # Shape: (k, k)
    
    # Compute the trace of M^T M
    trace_overlap = torch.trace(M.T @ M)
    
    if normalize:
        k = U_A.shape[1]  # Subspace dimension
        result = trace_overlap / k
    else:
        result = trace_overlap
    
    # Return the result on CPU
    return result.cpu()

def binet_cauchy_similarity(U_A, U_B, normalize=True, device='cuda'):
    """
    Compute the Binet-Cauchy similarity between two subspaces using the determinant relationship.
    Arguments:
        U_A: Tensor of shape (n, k), orthonormal basis of subspace A.
        U_B: Tensor of shape (n, k), orthonormal basis of subspace B.
    Returns:
        binet_cauchy_similarity: The Binet-Cauchy similarity between the subspaces.
    """
    # Compute the determinant of U_A^T U_B
    U_A = U_A.to(device)
    U_B = U_B.to(device)
    M = torch.matmul(U_A.T, U_B)  # Shape: (k, k)
    determinant = torch.det(M)
    k = U_A.shape[1]
    if normalize:
        result = (determinant**2)**(1/k)
    else:
        result = determinant**2

    return result.cpu()

def squared_cosine_similarity(U_A, U_B, device='cuda'):
    """
    Compute the cosine similarity between the top-k singular vectors of A and B.

    rguments:
        U_A: Tensor of shape (n, k), orthonormal basis of subspace A.
        U_B: Tensor of shape (n, k), orthonormal basis of subspace B.

    Returns:
        cosine_similarity: Average squared cosine similarity.
    """
    # Device
    U_A = U_A.to(device)
    U_B = U_B.to(device)

    # Compute cosine similarity for each vector
    similarities = torch.abs(torch.sum(U_A * U_B, dim=0))  # Dot product per column
    return torch.mean(similarities**2).cpu()  # Average squared cosine similarity

def absolute_cosine_similarity(U_A, U_B, device='cuda'):
    """
    Compute the cosine similarity between the top-k singular vectors of A and B.

    rguments:
        U_A: Tensor of shape (n, k), orthonormal basis of subspace A.
        U_B: Tensor of shape (n, k), orthonormal basis of subspace B.

    Returns:
        cosine_similarity: Average squared cosine similarity.
    """
    # Device
    U_A = U_A.to(device)
    U_B = U_B.to(device)

    # Compute cosine similarity for each vector
    similarities = torch.abs(torch.sum(U_A * U_B, dim=0))  # Dot product per column
    return torch.mean(similarities).cpu()  # Average absolute cosine similarity

def cosine_similarity(U_A, U_B, device='cuda'):
    """
    Compute the cosine similarity between the top-k singular vectors of A and B.

    rguments:
        U_A: Tensor of shape (n, k), orthonormal basis of subspace A.
        U_B: Tensor of shape (n, k), orthonormal basis of subspace B.

    Returns:
        cosine_similarity: Average squared cosine similarity.
    """
    # Device
    U_A = U_A.to(device)
    U_B = U_B.to(device)

    # Compute cosine similarity for each vector
    similarities = torch.abs(torch.sum(U_A * U_B, dim=0))  # Dot product per column
    return (torch.mean(similarities).cpu(), torch.mean(similarities**2).cpu())  # Average squared cosine similarity


def diag_sv_trace_similarity(J1, S1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ J1 @ V2
    tr = torch.trace(S1)
    norm = torch.linalg.norm(S1)
    diff = torch.linalg.norm(torch.abs(M)-S1)
    return diff / tr, diff / norm

# def diag_sv_norm_similarity(J1, S1, U2, V2): # swap U2 and V2 for the vju case
#     M = U2.T @ J1 @ V2
#     norm = torch.linalg.norm(S1)
#     return torch.linalg.norm(torch.abs(M)-S1) / norm

def diag_sv_similarity(U1, V1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ V1.T @ V2
    return torch.linalg.norm(torch.abs(M)-torch.eye(M.shape[0]))


def diag_sv_similarity_k(U1, S1, V1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ S1 @ V1.T @ V2
    tr = torch.trace(S1)
    norm = torch.linalg.norm(S1)
    diff = torch.linalg.norm(torch.abs(M)-S1)
    return diff / tr, diff / norm


def alignment_metric_new(Jac):
    #K = 30
    #K = 40
    diag_mat_all_k = {}
    alt_mat_all_k = {}
    alt_mat_sing_all_k = {}
    aln_uu_all_k = {}
    aln_vv_all_k = {}
    aln_uv_all_k = {}
    aln_vu_all_k = {}
    
    aln_ujv_all_k = {}
    aln_vju_all_k = {}

    L = 20
    E = 5
    ITS = 20
    percents = [0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    num_sing_vecs = [10, 30, 50]
    for percent in percents:
        num_sing_vecs.append(int(percent * Jac[0].shape[0]))

    #U_all, S, V_all = SVD(Jac, 300, L, E, ITS)
    U_all, S, V_all = svd(Jac) # nonrandom
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in Jac]
    
    for K in num_sing_vecs:
        U, V = [u[:, :K] for u in U_all], [v[ :K, :].T for v in V_all] #nonrandom
        #U, V = [u[..., :K] for u in U_all], [v[..., :K+E] for v in V_all]

        # uu_mat_proj = torch.zeros((len(S), len(S)))
        # vv_mat_proj = torch.zeros((len(S), len(S)))
        # uv_mat_proj = torch.zeros((len(S), len(S)))
        # vu_mat_proj = torch.zeros((len(S), len(S)))

        # uu_mat_binet = torch.zeros((len(S), len(S)))
        # vv_mat_binet = torch.zeros((len(S), len(S)))
        # uv_mat_binet = torch.zeros((len(S), len(S)))
        # vu_mat_binet = torch.zeros((len(S), len(S)))
        # uu_mat_cos_abs = torch.zeros((len(S), len(S)))
        # vv_mat_cos_abs = torch.zeros((len(S), len(S)))
        # uv_mat_cos_abs = torch.zeros((len(S), len(S)))
        # vu_mat_cos_abs = torch.zeros((len(S), len(S)))

        # uu_mat_cos_sq = torch.zeros((len(S), len(S)))
        # vv_mat_cos_sq = torch.zeros((len(S), len(S)))
        # uv_mat_cos_sq = torch.zeros((len(S), len(S)))
        # vu_mat_cos_sq = torch.zeros((len(S), len(S)))

        uuvv_mat = torch.zeros((len(S), len(S)))
        vuvu_mat = torch.zeros((len(S), len(S)))

        ujv_mat_trace = torch.zeros((len(S), len(S)))
        vju_mat_trace = torch.zeros((len(S), len(S)))

        ujv_mat_norm = torch.zeros((len(S), len(S)))
        vju_mat_norm = torch.zeros((len(S), len(S)))

        uusvv_mat_trace = torch.zeros((len(S), len(S)))
        vusvu_mat_trace = torch.zeros((len(S), len(S)))

        uusvv_mat_norm = torch.zeros((len(S), len(S)))
        vusvu_mat_norm = torch.zeros((len(S), len(S)))

        print(S[0].shape)


        for i in range(len(S)):
            for j in range(len(S)):
                uj, ji, vj = U[j], J[i], V[j]
                ui, vi = U[i], V[i]
                if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
                    print("wrong shape")
                    continue
                
                S_i = torch.diag(S[i][:K])

                ujv_mat_trace[i, j], ujv_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, uj, vj)
                vju_mat_trace[i, j], vju_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, vj, uj)

                # Diagonal SV Norm Similarity
                #ujv_mat_norm[i, j] = diag_sv_norm_similarity(ji, S_i, uj, vj)
                #vju_mat_norm[i, j] = diag_sv_norm_similarity(ji, S_i, vj, uj)

                # Diagonal SV Similarity
                uuvv_mat[i, j] = diag_sv_similarity(ui, vi, uj, vj)
                vuvu_mat[i, j] = diag_sv_similarity(ui, vi, vj, uj)

                # uusvv_mat_trace[i, j] = diag_sv_similarity_k(ui, S_i, vi, uj, vj, trace=True)
                # vusvu_mat_trace[i, j] = diag_sv_similarity_k(ui, S_i, vi, vj, uj, trace=True)
                uusvv_mat_trace[i, j], uusvv_mat_norm[i, j] = diag_sv_similarity_k(ui, S_i, vi, uj, vj)
                vusvu_mat_trace[i, j], vusvu_mat_norm[i, j] = diag_sv_similarity_k(ui, S_i, vi, vj, uj)

                # Projection
                # uu_proj = projection_similarity(uj, ui) 
                # vv_proj = projection_similarity(vj, vi) # Technically don't need to loop through everything since metric is symmetric

                # uv_proj = projection_similarity(uj, vi) # Only need uv since metric is symmetric
                # vu_proj = projection_similarity(vj, ui)

                
                # # Binet Cauchy
                # uu_binet = binet_cauchy_similarity(uj, ui)
                # vv_biet = binet_cauchy_similarity(vj, vi)
                # uv_binet = binet_cauchy_similarity(uj, vi)
                # vu_binet = binet_cauchy_similarity(vj, ui)
                # uu_cos_abs, uu_cos_sq = cosine_similarity(uj, ui)
                # vv_cos_abs, vv_cos_sq = cosine_similarity(vj, vi)
                # uv_cos_abs, uv_cos_sq = cosine_similarity(uj, vi)
                # vu_cos_abs, vu_cos_sq = cosine_similarity(vj, ui)

                # uu_mat_proj[i, j] = uu_proj
                # vv_mat_proj[i, j] = vv_proj
                # uv_mat_proj[i, j] = uv_proj
                # vu_mat_proj[i, j] = vu_proj

                # uu_mat_binet[i, j] = uu_binet
                # vv_mat_binet[i, j] = vv_biet
                # uv_mat_binet[i, j] = uv_binet
                # vu_mat_binet[i, j] = vu_binet

                # uu_mat_cos_abs[i, j] = uu_cos_abs
                # vv_mat_cos_abs[i, j] = vv_cos_abs
                # uv_mat_cos_abs[i, j] = uv_cos_abs
                # vu_mat_cos_abs[i, j] = vu_cos_abs

                # uu_mat_cos_sq[i, j] = uu_cos_sq
                # vv_mat_cos_sq[i, j] = vv_cos_sq
                # uv_mat_cos_sq[i, j] = uv_cos_sq
                # vu_mat_cos_sq[i, j] = vu_cos_sq


        aln_ujv_all = {}
        aln_ujv_all['trace'] = ujv_mat_trace
        aln_ujv_all['diag'] = uuvv_mat
        aln_ujv_all['norm'] = ujv_mat_norm
        
        aln_ujv_all['trace_k'] = uusvv_mat_trace
        aln_ujv_all['norm_k'] = uusvv_mat_norm

        aln_vju_all = {}
        aln_vju_all['trace'] = vju_mat_trace
        aln_vju_all['diag'] = vuvu_mat
        aln_vju_all['norm'] = vju_mat_norm

        aln_vju_all['trace_k'] = vusvu_mat_trace
        aln_vju_all['norm_k'] = vusvu_mat_norm

        aln_ujv_all_k[K] = aln_ujv_all
        aln_vju_all_k[K] = aln_vju_all

        # aln_uu_all = {}
        # # aln_uu_all['proj'] = uu_mat_proj
        # # aln_uu_all['binet'] = uu_binet
        # aln_uu_all['cos_abs'] = uu_mat_cos_abs
        # aln_uu_all['cos_sq'] = uu_mat_cos_sq

        # aln_uu_all_k[K] = aln_uu_all


        # aln_vv_all = {}
        # # aln_vv_all['proj'] = vv_mat_proj
        # # aln_vv_all['binet'] = vv_biet
        # aln_vv_all['cos_abs'] = vv_mat_cos_abs
        # aln_vv_all['cos_sq'] = vv_mat_cos_sq

        # aln_vv_all_k[K] = aln_vv_all

        # aln_uv_all = {}
        # # aln_uv_all['proj'] = uv_mat_proj
        # # aln_uv_all['binet'] = uv_binet
        # aln_uv_all['cos_abs'] = uv_mat_cos_abs
        # aln_uv_all['cos_sq'] = uv_mat_cos_sq

        # aln_uv_all_k[K] = aln_uv_all

        # aln_vu_all = {}
        # # aln_vu_all['proj'] = vu_mat_proj
        # # aln_vu_all['binet'] = vu_binet
        # aln_vu_all['cos_abs'] = vu_mat_cos_abs
        # aln_vu_all['cos_sq'] = vu_mat_cos_sq

        # aln_vu_all_k[K] = aln_vu_all


    return aln_ujv_all_k, aln_vju_all_k, aln_uu_all_k, aln_vv_all_k, aln_uv_all_k, aln_vu_all_k

def jacobian_eff(output, input, index, device, chunks):
    print(output.shape)
    #tt = time.time()
    
    output = output[0, index, :]
    N = output.numel()
    I_N = torch.eye(N).to(device)
    def get_vjp(v):
        return grad(output, input, v, retain_graph=True)[0][0, index, :]
        

    jacobian = chunk_vmap(get_vjp, chunks=chunks)(I_N)

    #print(time.time()-tt)

    return jacobian


def jacobians_llama(num_layers, outputs, k):
  token = k-1

  Jac_noskip = []
  #print(tokenizer.decode(input_ids[0, token+1]))
  for i in range(num_layers-1): ### -1 is to not count the last layer
    j = jacobian(outputs.hidden_states[i+1], outputs.hidden_states[i], 1+token, 'cuda').cpu().float()
    dim = j.shape[0]
    Jac_noskip.append(j - torch.eye(dim))
  if token == k-1:
    label = 'final'
  else:
    label = token

  return Jac_noskip, label

def jacobians_falcon(num_layers, outputs, k):
  token = k-1

  Jac_noskip = []
  #print(tokenizer.decode(input_ids[0, token]))
  for i in range(num_layers-1): ### -1 is to not count the last layer
    j = jacobian(outputs.hidden_states[i+1], outputs.hidden_states[i], token, 'cuda').cpu().float()
    dim = j.shape[0]
    Jac_noskip.append(j - torch.eye(dim))
  if token == k-1:
    label = 'final'
  else:
    label = token

  return Jac_noskip, label

def jacobians(num_layers, outputs, k, chunks):
  token = k

  Jac_noskip = []
  #print(tokenizer.decode(input_ids[0, token]))
  
  for i in range(num_layers-1):
    j = jacobian_eff(outputs.hidden_states[i+1], outputs.hidden_states[i], token, 'cuda', chunks).cpu().float()
    dim = j.shape[0]
    Jac_noskip.append(j - torch.eye(dim))
  if token == -1:
    label = 'final'
  else:
    label = token

  return Jac_noskip, label

def compute_alignment(model, input_ids, chunks):
    tt = time.time()
    outputs = model(input_ids, output_hidden_states=True)
    k = -1
    print('Memory info (after forward):', torch.cuda.mem_get_info())

    num_layers = len(outputs.hidden_states) - 1
    Jac_noskip, label = jacobians(num_layers, outputs, k, chunks)
    diag_mat, frob_mat, frob_mat_sing, uu_mat, vv_mat, uv_mat, vu_mat = alignment_metric(Jac_noskip)
    print('Compute alignment time:', time.time()-tt)
    
    #outputs.hidden_states.cpu()
    #del outputs
    del outputs
    gc.collect()
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return diag_mat, frob_mat, frob_mat_sing, uu_mat, vv_mat, uv_mat, vu_mat

def compute_alignment_new(model, input_ids, chunks):
    tt = time.time()
    outputs = model(input_ids, output_hidden_states=True)
    k = -1
    print('Memory info (after forward):', torch.cuda.mem_get_info())

    num_layers = len(outputs.hidden_states) - 1
    Jac_noskip, label = jacobians(num_layers, outputs, k, chunks)
    ujv_mat, vju_mat, uu_mat, vv_mat, uv_mat, vu_mat = alignment_metric_new(Jac_noskip)
    print('Compute alignment time:', time.time()-tt)
    
    #outputs.hidden_states.cpu()
    #del outputs
    del outputs
    gc.collect()
    #with torch.no_grad():
    #    torch.cuda.empty_cache()
    torch.cuda.empty_cache()

    return ujv_mat, vju_mat, uu_mat, vv_mat, uv_mat, vu_mat

def compute_linearization(model, input_ids, chunks):
    tt = time.time()
    outputs = model(input_ids, output_hidden_states=True)
    k = -1
    print('Memory info (after forward):', torch.cuda.mem_get_info())

    num_layers = len(outputs.hidden_states) - 1
    Jac_noskip, label = jacobians(num_layers, outputs, k, chunks)

    print('Compute Jacobians time:', time.time()-tt)
    t = time.time()-tt

    x_final = outputs.hidden_states[-1][0, k].to("cuda")
    x_lin = outputs.hidden_states[0][0, k].to("cuda")

    dt = x_lin.dtype

    Id = torch.eye(Jac_noskip[0].shape[0], dtype=dt).to("cuda")
    for J in Jac_noskip:
        J = J.detach().to("cuda").to(dt)
        x_lin = (Id + J) @ x_lin

    c = torch.sum(x_final * x_lin) / (torch.norm(x_lin) * torch.norm(x_final))
    return c.cpu(), t

    




