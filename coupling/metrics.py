import torch

from .jacobian import svd

def metrics(Jac, Us=None, Ss=None, Vs=None, p=2, num_sing_vecs=(10,30,50), 
    svd_method="torch", L=20, E=5, ITS=20, device="cpu", verbose=False):
    """
    Main method for computing coupling metrics between layer-wise Jacobians.

    Jac:            Jacobians across skip connection
    Us, Ss, Vs:     SVDs of Jac
    - if None, they are computed within this function
    p:              order of p-norm for coupling measurement
    num_sing_vecs:  number of top singular vectors to use in computing coupling 
    svd_method:     method for computing svd, see `jacobian.svd`
    - if using "random", `K, L, E, ITS` will be used
    """
    
    aln_ujv_all_k = {}
    aln_vju_all_k = {}

    if Us is None or Ss is None or Vs is None:
        Us, Ss, Vs = svd(Jac, K=max(num_sing_vecs), L=L, E=E, ITS=ITS, method=svd_method, verbose=verbose)

    S = torch.stack(Ss).cpu()
    U_all = [u.to(device) for u in Us]
    V_all = [v.to(device) for v in Vs]
    J = [j.to(device) for j in Jac]

    for K in num_sing_vecs:
        U, V = [u[:, :K] for u in U_all], [v[ :K, :].T for v in V_all]

        ujv_mat_trace = torch.zeros((len(S), len(S)))
        vju_mat_trace = torch.zeros((len(S), len(S)))

        ujv_mat_norm = torch.zeros((len(S), len(S)))
        vju_mat_norm = torch.zeros((len(S), len(S)))

        for i in range(len(S)):
            for j in range(len(S)):
                uj, ji, vj = U[j], J[i], V[j]
                # ui, vi = U[i], V[i]
                if uj.shape[0] != ji.shape[1] or vj.shape[0] != ji.shape[0]:
                    print("wrong shape")
                    continue
                
                # S[i] 1D, S_i 2D
                S_i = torch.diag(S[i][:K])

                ujv_mat_trace[i, j], ujv_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, uj, vj, p=p)
                vju_mat_trace[i, j], vju_mat_norm[i, j] = diag_sv_trace_similarity(ji, S_i, vj, uj, p=p)

        aln_ujv_all = {}
        aln_ujv_all['trace'] = ujv_mat_trace
        aln_ujv_all['norm'] = ujv_mat_norm

        aln_vju_all = {}
        aln_vju_all['trace'] = vju_mat_trace
        aln_vju_all['norm'] = vju_mat_norm

        aln_ujv_all_k[K] = aln_ujv_all
        aln_vju_all_k[K] = aln_vju_all

    return aln_ujv_all_k, aln_vju_all_k

def diag_sv_trace_similarity(J1, S1, U2, V2, p=2): # swap U2 and V2 for the vju case
    """ 
    TODO: Main coupling metric
    """
    M = U2.T @ J1 @ V2
    tr = torch.trace(S1)
    norm = torch.norm(torch.diag(S1), p=p)
    diff = torch.linalg.norm(torch.abs(M)-S1)
    return diff / tr, diff / norm

def diag_sv_similarity(U1, V1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ V1.T @ V2
    return torch.linalg.norm(torch.abs(M)-torch.eye(M.shape[0]))

def diag_sv_similarity_k(U1, S1, V1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ U1 @ S1 @ V1.T @ V2
    tr = torch.trace(S1)
    norm = torch.linalg.norm(S1)
    diff = torch.linalg.norm(torch.abs(M)-S1)
    return diff / tr, diff / norm
