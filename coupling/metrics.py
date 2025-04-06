import torch

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

def metrics(Jac, num_sing_vecs=(10,30,50)):
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
    #percents = [0.005, 0.008, 0.01, 0.02, 0.03, 0.04, 0.05, 0.1]
    #for percent in percents:
    #    num_sing_vecs.append(int(percent * Jac[0].shape[0]))

    #U_all, S, V_all = SVD(Jac, 300, L, E, ITS)
    U_all, S, V_all = svd(Jac) # nonrandom
    S = torch.stack(S).cpu()
    J = [j.cpu() for j in Jac]
    
    for K in num_sing_vecs:
        U, V = [u[:, :K] for u in U_all], [v[ :K, :].T for v in V_all]

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

                uusvv_mat_trace[i, j], uusvv_mat_norm[i, j] = diag_sv_similarity_k(ui, S_i, vi, uj, vj)
                vusvu_mat_trace[i, j], vusvu_mat_norm[i, j] = diag_sv_similarity_k(ui, S_i, vi, vj, uj)


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

    return aln_ujv_all_k, aln_vju_all_k, aln_uu_all_k, aln_vv_all_k, aln_uv_all_k, aln_vu_all_k

def diag_sv_trace_similarity(J1, S1, U2, V2): # swap U2 and V2 for the vju case
    M = U2.T @ J1 @ V2
    tr = torch.trace(S1)
    norm = torch.linalg.norm(S1)
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
