import time
import gc

import torch
from torch.autograd import grad
from functorch.experimental import chunk_vmap

from .utils import timestamp

def jacobian(output, input, index, chunks, index_in=None, verbose=False, device="cuda"):
    """
    Computes the Jacobian of `d{output}/d{input}` from transformer hooks
    by vectorizing over gradients.

    output:     Jacobian wrt this output
    input:      Jacobian wrt this input
    index:      index of output token
    chunks:     number of chunks used to vectorize Jacobian computation
    index_in:   (optional) changes input token of Jacobian if not `index`
    """
    tt = time.time()

    output = output[0, index, :]
    I_N = torch.eye(output.numel()).to(device)

    index_in = index_in if index_in is not None else index
    def get_vjp(v):
        return grad(output, input, v, retain_graph=True)[0][0, index_in, :]

    jacobian = chunk_vmap(get_vjp, chunks=chunks)(I_N)
    print(f"Jacobian computed in {time.time()-tt:.3f} seconds") if verbose else None

    return jacobian

def svd(Jac, method="torch", K=50, L=20, E=5, ITS=20, verbose=False):
    """
    Computes the SVD of a list of Jacobians.

    Jac:        list of Jacobians
    method:     
    -   "torch" calls `torch.linalg.svd()`
    -   "random" calls random SVD (implemented below),
        - if using "random", `K, L, E, ITS` will be used
    """
    if method == "torch":
        Us, Ss, Vs = [], [], []
        for j in range(len(Jac)):
            i = j
            tt = time.time()
            timestamp('svding layer {}'.format(i))
            U, S, V = torch.linalg.svd(Jac[i])
            Us.append(U)
            Ss.append(S)
            Vs.append(V)
            print(f"SVD computed in {time.time()-tt:.3f} seconds") if verbose else None
    elif method == "random":
        Us, Ss, Vs = randomsvd(Jac, K=K, L=L, E=E, ITS=ITS)
    else:
        raise NotImplementedError("Only torch SVD is implemented")
    return Us, Ss, Vs

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