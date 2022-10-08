import numpy as np
import torch
import torch.utils.data
from scipy.stats import chi

is_cuda = True
dtype = torch.float
device = torch.device("cuda:0")
#i don't know what it does
class ModelLatentF(torch.nn.Module):
    """define deep networks."""
    def __init__(self, x_in, H, x_out):
        """Init latent features."""
        super(ModelLatentF, self).__init__()
        self.restored = False

        self.latent = torch.nn.Sequential(
            torch.nn.Linear(x_in, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bixas=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, H, bias=True),
            torch.nn.Softplus(),
            torch.nn.Linear(H, x_out, bias=True),
        )
    def forward(self, input):
        """Forward the LeNet."""
        fealant = self.latent(input)
        return fealant

    
def mem_ratio():
    print(torch.cuda.memory_allocated() / torch.cuda.memory_reserved())

def get_item(x, is_cuda):
    """get the numpy value from a torch tensor."""
    if is_cuda:
        x = x.cpu().detach().numpy()
    else:
        x = x.detach().numpy()
    return x

def MatConvert(x, device, dtype):
    """convert the numpy to a torch tensor."""
    x = torch.from_numpy(x).to(device, dtype)
    return x

def Pdist2(x, y):
    """compute the paired distance between x and y."""
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)
    Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    Pdist[Pdist<0]=0
    return Pdist

#returns all the pairs for n integers with a (n*n)*2 shaped numpy ndarray
def get_pairs(n):
    return np.array(np.meshgrid(np.arange(n), np.arange(n))).T.reshape(-1, 2)

# gets as input the kernel values , k(x,x'),k(y,y'), k(x,y) and computes mmd and estimate for variance
### todo: this function should be changed so that it can accept array of kerenl values and returns array of mmd and corresponding covariance matrix and it should also support gamma look maybe mmd_multi already does this
def h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, gamma=2):
    """compute value of MMD and std of MMD using kernel matrix."""
    Kxxy = torch.cat((Kx,Kxy),1)
    Kyxy = torch.cat((Kxy.transpose(0,1),Ky),1)
    Kxyxy = torch.cat((Kxxy,Kyxy),0)
    n = nx = Kx.shape[0]
    ny = Ky.shape[0]
    # N = torch.zeros(n, n, dtype=dtype)
    # D = get_pairs(n)
    # idx0, idx1 = D[:,0], D[:,1]
    # #import IPython; IPython.embed()
    # x1, x2 = Kx[idx0], Kx[idx1]
    # y1, y2 = Ky[idx0], Ky[idx1]
    # l = len(x1)
    # if self.gamma == 1.:
    #     # # for linear time mmd assume that the number of samples is 2n. Truncate last data point if uneven
    #     l = n2 = int(n / 2)
    #     idx0=np.array(range(n2))
    #     idx1=n2+np.array(range(n2))
    #     D = np.zeros((2,n2))
    #     D[0]=idx0
    #     D[1]=idx1
    #     D = D.reshape(n2,2)
    #     N[np.arange(n2), np.arange(n2) + n2] += 1
    #     #for i in range(n2):
    #     #    N[i,i+n2] +=1
    # else:
    #     if self.gamma == 2.:
    #         N = torch.ones(n,n)-torch.eye(n)
    #     else:
    #         l = subset_size = int(round(0.5*(size**self.gamma-(self.gamma-1)*size),0))
    #         D = D[np.random.choice(range(len(D)), subset_size, replace=True)]
    #         N[D[:,0], D[:,1]] += 1
    
    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    is_unbiased = True
    if is_unbiased:
        xx = torch.div((torch.sum(Kx) - torch.sum(torch.diag(Kx))), (nx * (nx - 1)))
        yy = torch.div((torch.sum(Ky) - torch.sum(torch.diag(Ky))), (ny * (ny - 1)))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy) - torch.sum(torch.diag(Kxy))), (nx * (ny - 1)))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    else:
        xx = torch.div((torch.sum(Kx)), (nx * nx))
        yy = torch.div((torch.sum(Ky)), (ny * ny))
        # one-sample U-statistic.
        if use_1sample_U:
            xy = torch.div((torch.sum(Kxy)), (nx * ny))
        else:
            xy = torch.div(torch.sum(Kxy), (nx * ny))
        mmd2 = xx - 2 * xy + yy
    if not is_var_computed:
        return mmd2, None

    hh = Kx+Ky-Kxy-Kxy.transpose(0,1)
    V1 = torch.dot(hh.sum(1)/ny,hh.sum(1)/ny) / ny
    V2 = (hh).sum() / (nx) / nx
    varEst = 4*(V1 - V2**2)
    if  varEst == 0.0:
        print('error!!'+str(V1))
    return mmd2, varEst, Kxyxy

def get_l(n,gamma):
    # return int(round(0.5*(n**gamma-(gamma-1)*n),0))
    return int(round(n**gamma,0))

def get_weights(n, gamma):
    I = np.eye(n)
    if gamma == 2:
        N = torch.ones(n,n)-I
    elif gamma == 1:
        n2 = int(n / 2)
        N = torch.zeros(n, n, dtype=dtype)
        N[np.arange(n2), np.arange(n2) + n2] += 1
    else:
        l = get_l(n,gamma)
        D = get_pairs(n)
        D = D[np.random.choice(range(len(D)), l, replace=False)]
        N = torch.zeros(n, n, dtype=dtype)
        N[D[:,0], D[:,1]] += 1
        N = N-np.diag(np.diag(N))
    
    W = N/torch.sum(N)
    return W


def get_coeff(n, gamma):
    l = get_l(n,gamma)
    A1 = (1 - 1.0 / l) * (4 * (n-2)) / (n * (n-1))
    A2 = (1 - 1.0 / l) * 2 / (n * (n-1)) + 1.0 / l
    A = A1 / A2
    if gamma==1:
        return 0.0, 2./n
    elif gamma==2:
        return 4./n, 2./(n**2)
    else:
        return A/l,1./l

#returns an n*n matrix which is the indicator of selected pairs devided by number of pairs
# no of pairs is [(n**gamma-n(gamma-1))/2]

def h1_mean_var_gram_multi(Kx, Ky, Kxy, is_var_computed, use_1sample_U=True, gamma=2):
    """compute value of MMD and std of MMD using kernel matrix."""
    n = Kx.shape[1]
    Kx = Kx.reshape(-1,n,n)
    Ky = Ky.reshape(-1,n,n)
    Kxy = Kxy.reshape(-1,n,n)
    num_kernels = Kx.shape[0]
    p = n*(n-1)
    pr=n**2
    W = get_weights(n, gamma)
    # print(torch.sum(W))
    H = torch.zeros(num_kernels, n, n)
    # import IPython; IPython.embed()
    H_bar = torch.zeros(num_kernels, n, n)
    one = torch.ones(n)
    means = torch.zeros(num_kernels, n)
    #gram are kx,ky,kxy
    mmd = torch.zeros(num_kernels)
    KKxyxy = torch.zeros(num_kernels,2*n,2*n)
    print("inside gram", mem_ratio())
    for u in range(num_kernels):
        print("loop gram", u, mem_ratio())
        Kx_bar = Kx[u]*W
        Ky_bar = Ky[u]*W
        Kxy_bar = Kxy[u]*W
        Kyx_bar = Kxy[u].transpose(0,1)*W
        Kxxy = torch.cat((Kx_bar,Kxy_bar),1)
        Kyxy = torch.cat((Kyx_bar,Ky_bar),1)
        Kxyxy = torch.cat((Kxxy,Kyxy),0)
        KKxyxy[u] = Kxyxy
        H[u]=hh = Kx[u]+Ky[u]-Kxy[u]-Kxy[u].transpose(0,1)
        H[u].fill_diagonal_(0)
        H_bar[u] = Kx_bar + Ky_bar - Kxy_bar - Kyx_bar
        H_bar[u].fill_diagonal_(0)
        
        # import IPython; IPython.embed()
        # is_unbiased = True
        # if is_unbiased:
        #     xx = torch.div((torch.sum(Kx[ii]) - torch.sum(torch.diag(Kx[ii]))), (n * (n - 1)))
        #     yy = torch.div((torch.sum(Ky[ii]) - torch.sum(torch.diag(Ky[ii]))), (n * (n - 1)))
        #     # one-sample U-statistic.
        #     if use_1sample_U:
        #         xy = torch.div((torch.sum(Kxy[ii]) - torch.sum(torch.diag(Kxy[ii]))), (n * (n - 1)))
        #     else:
        #         xy = torch.div(torch.sum(Kxy[ii]), (n * n))
        #     mmd[i] = xx - 2 * xy + yy
        # else:
        # xx = torch.div((torch.sum(Kx[u])), (pr))
        # yy = torch.div((torch.sum(Ky[u])), (pr))
        # one-sample U-statistic.
        # if use_1sample_U:
        #     xy = torch.div((torch.sum(Kxy[u])), (nx * ny))
        # else:
        # xy = torch.div(torch.sum(Kxy[u]), (pr))
        # mmd[u] = xx - 2 * xy + yy
        # mmd_c
    print("outside gram", mem_ratio())
    mmd = torch.sum(H_bar,(1,2))
    mmd_c = torch.sum(H, (1,2))/p
    means = torch.sum(H,2)/n    
    if not is_var_computed:
        return mmd, None
    # S1 = torch.dot(hh.sum(1)/n,hh.sum(1)/n) / n - ((hh).sum() / (n) / n)**2
    # S1 = 1 / n * means.matmul(means.transpose(0, 1)) - mmd_c.view(-1, 1).matmul(mmd_c.view(1, -1))
    S1 = 1 / n * means.matmul(means.transpose(0, 1)) - mmd_c.outer(mmd_c)
    # S2 = 1 / p * (H.reshape(num_kernels, -1)).matmul((H.reshape(num_kernels, -1)).transpose(0, 1)) - mmd_c.view(-1, 1).matmul(mmd_c.view(1, -1))
    S2 = 1 / p * (H.reshape(num_kernels, -1)).matmul((H.reshape(num_kernels, -1)).transpose(0, 1)) - mmd_c.outer(mmd_c)
    # 100 x 200 x 200 
    A,B = get_coeff(n, gamma)
    V = A*S1+B*S2
    # if  varEst == 0.0:
    #     print('error!!'+str(V1))
    return mmd, V, KKxyxy


def MMDu(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon=10 ** (-10), is_smooth=True, is_var_computed=True, use_1sample_U=True, gamma=2):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    L = 1 # generalized Gaussian (if L>1)
    Dxx = Pdist2(X, X)
    Dyy = Pdist2(Y, Y)
    Dxy = Pdist2(X, Y)
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    if is_smooth:
        Kx = (1.0-epsilon) * torch.exp(-(Dxx / sigma0) - (Dxx_org / sigma)) + epsilon * torch.exp(-Dxx_org / sigma)
        Ky = (1.0-epsilon) * torch.exp(-(Dyy / sigma0) - (Dyy_org / sigma)) + epsilon * torch.exp(-Dyy_org / sigma)
        Kxy = (1.0-epsilon) * torch.exp(-(Dxy / sigma0) - (Dxy_org / sigma)) + epsilon * torch.exp(-Dxy_org / sigma)
    else:
        Kx = torch.exp(-Dxx / sigma0)
        Ky = torch.exp(-Dyy / sigma0)
        Kxy = torch.exp(-Dxy / sigma0)
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U,gamma)

def MMDg(Fea, len_s, Fea_org, sigma, sigma0, epsilon, is_smooth=True, is_var_computed=True, use_1sample_U=True, gamma=2):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    num_kernels = len(Fea)
    # X = Fea[0:len_s, :] # fetch the sample 1 (features of deep networks)
    # Y = Fea[len_s:, :] # fetch the sample 2 (features of deep networks)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    L = 1 # generalized Gaussian (if L>1)
    # Dxx = Pdist2(X, X)
    # Dyy = Pdist2(Y, Y)
    # Dxy = Pdist2(X, Y)
    # print(num_kernels)
    # print(len_s)
    KKx=torch.zeros(num_kernels,len_s,len_s,dtype=dtype, device = 'cpu')
    KKy=torch.zeros(num_kernels,len_s,len_s,dtype=dtype,device = 'cpu')
    KKxy=torch.zeros(num_kernels,len_s,len_s,dtype=dtype,device = 'cpu')
    print("inside mmdg", mem_ratio())
    for ii in range(num_kernels):
        print("loop mmdg", ii, mem_ratio())
        X = Fea[ii][0:len_s, :]  # fetch the sample 1 (features of deep networks)
        Y = Fea[ii][len_s:, :]  # fetch the sample 2 (features of deep networks)
        Dxx = Pdist2(X, X)
        Dyy = Pdist2(Y, Y)
        Dxy = Pdist2(X, Y)

        if is_smooth:
            Kx = (1.0-epsilon[ii]) * torch.exp(-(Dxx / sigma0[ii]) - (Dxx_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxx_org / sigma[ii])
            Ky = (1.0-epsilon[ii]) * torch.exp(-(Dyy / sigma0[ii]) - (Dyy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dyy_org / sigma[ii])
            Kxy = (1.0-epsilon[ii]) * torch.exp(-(Dxy / sigma0[ii]) - (Dxy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxy_org / sigma[ii])
        else:
            Kx = torch.exp(-Dxx / sigma0[ii])
            Ky = torch.exp(-Dyy / sigma0[ii])
            Kxy = torch.exp(-Dxy / sigma0[ii])
        KKx[ii]=Kx
        KKy[ii]=Ky
        KKxy[ii]=Kxy
    print("outside mmdg",mem_ratio())    
    return h1_mean_var_gram_multi(KKx, KKy, KKxy, is_var_computed, use_1sample_U,gamma)

def MMDu_multi(weights, Fea, len_s, Fea_org, sigma, sigma0, epsilon, is_smooth=True, is_var_computed=True, use_1sample_U=True,gamma=2):
    """compute value of deep-kernel MMD and std of deep-kernel MMD using merged data."""
    num_kernels = len(Fea)
    X_org = Fea_org[0:len_s, :] # fetch the original sample 1
    Y_org = Fea_org[len_s:, :] # fetch the original sample 2
    Dxx_org = Pdist2(X_org, X_org)
    Dyy_org = Pdist2(Y_org, Y_org)
    Dxy_org = Pdist2(X_org, Y_org)
    for ii in range(num_kernels):
        X = Fea[ii][0:len_s, :]  # fetch the sample 1 (features of deep networks)
        Y = Fea[ii][len_s:, :]  # fetch the sample 2 (features of deep networks)
        Dxx = Pdist2(X, X)
        Dyy = Pdist2(Y, Y)
        Dxy = Pdist2(X, Y)

        if is_smooth:
            Kx = (1.0-epsilon[ii]) * torch.exp(-(Dxx / sigma0[ii]) - (Dxx_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxx_org / sigma[ii])
            Ky = (1.0-epsilon[ii]) * torch.exp(-(Dyy / sigma0[ii]) - (Dyy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dyy_org / sigma[ii])
            Kxy = (1.0-epsilon[ii]) * torch.exp(-(Dxy / sigma0[ii]) - (Dxy_org / sigma[ii])) + epsilon[ii] * torch.exp(-Dxy_org / sigma[ii])
        else:
            Kx = torch.exp(-Dxx / sigma0[ii])
            Ky = torch.exp(-Dyy / sigma0[ii])
            Kxy = torch.exp(-Dxy / sigma0[ii])

        if ii == 0:
            Kx_all = weights[ii] * Kx
            Ky_all = weights[ii] * Ky
            Kxy_all = weights[ii] * Kxy
        else:
            Kx_all = Kx_all + weights[ii] * Kx
            Ky_all = Ky_all + weights[ii] * Ky
            Kxy_all = Kxy_all + weights[ii] * Kxy

    return h1_mean_var_gram_multi(Kx_all, Ky_all, Kxy_all, is_var_computed, use_1sample_U,gamma)

def MMDu_linear_kernel(Fea, len_s, is_var_computed=True, use_1sample_U=True,gamma=2):
    """compute value of (deep) lineaer-kernel MMD and std of (deep) lineaer-kernel MMD using merged data."""
    try:
        X = Fea[0:len_s, :]
        Y = Fea[len_s:, :]
    except:
        X = Fea[0:len_s].unsqueeze(1)
        Y = Fea[len_s:].unsqueeze(1)
    Kx = X.mm(X.transpose(0,1))
    Ky = Y.mm(Y.transpose(0,1))
    Kxy = X.mm(Y.transpose(0,1))
    return h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed, use_1sample_U,gamma)

def mmd2_permutations(K, n_X, permutations=500):
    """
        Fast implementation of permutations using kernel matrix.
    """
    K = torch.as_tensor(K)
    n = K.shape[0]
    assert K.shape[0] == K.shape[1]
    n_Y = n_X
    assert n == n_X + n_Y
    w_X = 1
    w_Y = -1
    ws = torch.full((permutations + 1, n), w_Y, dtype=K.dtype, device=K.device)
    ws[-1, :n_X] = w_X
    for i in range(permutations):
        ws[i, torch.randperm(n)[:n_X].numpy()] = w_X
    biased_ests = torch.einsum("pi,ij,pj->p", ws, K, ws)
    if True:  # u-stat estimator
        # need to subtract \sum_i k(X_i, X_i) + k(Y_i, Y_i) + 2 k(X_i, Y_i)
        # first two are just trace, but last is harder:
        is_X = ws > 0
        X_inds = is_X.nonzero()[:, 1].view(permutations + 1, n_X)
        Y_inds = (~is_X).nonzero()[:, 1].view(permutations + 1, n_Y)
        del is_X, ws
        cross_terms = K.take(Y_inds * n + X_inds).sum(1)
        del X_inds, Y_inds
        ests = (biased_ests - K.trace() + 2 * cross_terms) / (n_X * (n_X - 1))
    est = ests[-1]
    rest = ests[:-1]
    p_val = (rest > est).float().mean()
    return est.item(), p_val.item(), rest

def TST_MMD_adaptive_bandwidth(Fea, N_per, N1, Fea_org, sigma, sigma0, alpha, device, dtype):
    """run two-sample test (TST) using ordinary Gaussian kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, is_smooth=False)
    mmd_value = get_item(TEMP[0],is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]

        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        #        print(np.int(np.ceil(N_per*alpha)))
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def Analytic_Weights(mmd, V):
    l = mmd.shape[0]
    epsilon=1e-5
    I = torch.eye(l,dtype=dtype, device = device)
    V = V.to(I)
    mmd = mmd.to(I)
    L = torch.linalg.cholesky(V+epsilon*I)
    beta = torch.cholesky_solve(mmd.reshape(-1,1), L)
    return beta.to(I)

def get_Analytic_Weights(Fea, N1, Fea_org, all_sigma, all_sigma0, all_ep, device, dtype, gamma=2, is_smooth=True):
    TEMP = MMDg(Fea, N1, Fea_org, all_sigma, all_sigma0, all_ep, device, dtype, gamma=gamma)
    weights = Analytic_Weights(TEMP[0], TEMP[1])
    return weights
    # return TEMP

def TST_MMD_u(Fea, N_per, N1, Fea_org, sigma, sigma0, epsilon, alpha, device, dtype, gamma=2, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu(Fea, N1, Fea_org, sigma, sigma0, epsilon,is_smooth,gamma)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h,threshold,mmd_value_nn

def TST_MMD_Multi(weights, Fea, N_per, N1, Fea_org, all_sigma, all_sigma0, all_ep, alpha, device, dtype, gamma=2, is_smooth=True):
    """run two-sample test (TST) using deep kernel kernel."""
    TEMP = MMDu_multi(weights, Fea, N1, Fea_org, all_sigma, all_sigma0, all_ep, gamma=gamma)
    Kxyxy = TEMP[2].to(device)
    s = Kxyxy.shape[1]
    Kxyxy = Kxyxy.reshape(s,s)
    nx = N1
    mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
    if p_val > alpha:
        h = 0
    else:
        h = 1
    threshold = "NaN"
    return h,threshold,mmd_value_nn

def TST_Wald(weights, Fea, N_per, N1, Fea_org, all_sigma, all_sigma0, all_ep, alpha, device, dtype, gamma=2, is_smooth=True):
    TEMP = MMDu_multi(weights, Fea, N1, Fea_org, all_sigma, all_sigma0, all_ep, gamma)
    t_obs = np.sqrt(TEMP[0])    
    threshold = chi.ppf(q=1-alpha, df=d)
    if t_obs > threshold:
        h = 1
    else:
        h = 0
    threshold = "NaN"
    return h,threshold,mmd_value_nn

# def TST_MMD_General(Fea, N_per, N1, Fea_org, all_sigma, all_sigma0, all_ep, alpha, device, dtype, gamma=2, is_smooth=True):
#     """run two-sample test (TST) using deep kernel kernel."""
#     TEMP = MMDg(Fea, N1, Fea_org, all_sigma, all_sigma0, all_ep)
#     Kxyxy = TEMP[2]
#     nx = N1
#     mmd_value_nn, p_val, rest = mmd2_permutations(Kxyxy, nx, permutations=200)
#     if p_val > alpha:
#         h = 0
#     else:
#         h = 1
#     threshold = "NaN"
#     return h,threshold,mmd_value_nn

def TST_MMD_u_linear_kernel(Fea, N_per, N1, alpha, device, dtype):
    """run two-sample test (TST) using (deep) lineaer kernel kernel."""
    mmd_vector = np.zeros(N_per)
    TEMP = MMDu_linear_kernel(Fea, N1)
    mmd_value = get_item(TEMP[0], is_cuda)
    Kxyxy = TEMP[2]
    count = 0
    nxy = Fea.shape[0]
    nx = N1
    for r in range(N_per):
        # print r
        ind = np.random.choice(nxy, nxy, replace=False)
        # divide into new X, Y
        indx = ind[:nx]
        # print(indx)
        indy = ind[nx:]
        Kx = Kxyxy[np.ix_(indx, indx)]
        # print(Kx)
        Ky = Kxyxy[np.ix_(indy, indy)]
        Kxy = Kxyxy[np.ix_(indx, indy)]
        TEMP = h1_mean_var_gram(Kx, Ky, Kxy, is_var_computed=False)
        mmd_vector[r] = TEMP[0]
        if mmd_vector[r] > mmd_value:
            count = count + 1
        if count > np.ceil(N_per * alpha):
            h = 0
            threshold = "NaN"
            break
        else:
            h = 1
    if h == 1:
        S_mmd_vector = np.sort(mmd_vector)
        threshold = S_mmd_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    return h, threshold, mmd_value.item()

def C2ST_NN_fit(S,y,N1,x_in,H,x_out,learning_rate_C2ST,N_epoch,batch_size,device,dtype):
    """Train a deep network for C2STs."""
    N = S.shape[0]
    if is_cuda:
        model_C2ST = ModelLatentF(x_in, H, x_out).cuda()
    else:
        model_C2ST = ModelLatentF(x_in, H, x_out)
    w_C2ST = torch.randn([x_out, 2]).to(device, dtype)
    b_C2ST = torch.randn([1, 2]).to(device, dtype)
    w_C2ST.requires_grad = True
    b_C2ST.requires_grad = True
    optimizer_C2ST = torch.optim.Adam(list(model_C2ST.parameters()) + [w_C2ST] + [b_C2ST], lr=learning_rate_C2ST)
    criterion = torch.nn.CrossEntropyLoss()
    f = torch.nn.Softmax()
    ind = np.random.choice(N, N, replace=False)
    tr_ind = ind[:np.int(np.ceil(N * 1))]
    te_ind = tr_ind
    dataset = torch.utils.data.TensorDataset(S[tr_ind,:], y[tr_ind])
    dataloader_C2ST = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    len_dataloader = len(dataloader_C2ST)
    for epoch in range(N_epoch):
        data_iter = iter(dataloader_C2ST)
        tt = 0
        while tt < len_dataloader:
            # training model using source data
            data_source = data_iter.next()
            S_b, y_b = data_source
            output_b = model_C2ST(S_b).mm(w_C2ST) + b_C2ST
            loss_C2ST = criterion(output_b, y_b)
            optimizer_C2ST.zero_grad()
            loss_C2ST.backward(retain_graph=True)
            # Update sigma0 using gradient descent
            optimizer_C2ST.step()
            tt = tt + 1
        if epoch % 100 == 0:
            print(criterion(model_C2ST(S).mm(w_C2ST) + b_C2ST, y).item())
    output = f(model_C2ST(S[te_ind,:]).mm(w_C2ST) + b_C2ST)
    pred = output.max(1, keepdim=True)[1]
    STAT_C2ST = abs(pred[:N1].type(torch.FloatTensor).mean() - pred[N1:].type(torch.FloatTensor).mean())
    return pred, STAT_C2ST, model_C2ST, w_C2ST, b_C2ST

def TST_C2ST(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
    """run C2ST-S."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    pred_C2ST = output.max(1, keepdim=True)[1]
    STAT = abs(pred_C2ST[:N1].type(torch.FloatTensor).mean() - pred_C2ST[N1:].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(pred_C2ST[ind_X].type(torch.FloatTensor).mean() - pred_C2ST[ind_Y].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT

def TST_LCE(S,N1,N_per,alpha,model_C2ST, w_C2ST, b_C2ST,device,dtype):
    """run C2ST-L."""
    np.random.seed(seed=1102)
    torch.manual_seed(1102)
    torch.cuda.manual_seed(1102)
    N = S.shape[0]
    f = torch.nn.Softmax()
    output = f(model_C2ST(S).mm(w_C2ST) + b_C2ST)
    STAT = abs(output[:N1,0].type(torch.FloatTensor).mean() - output[N1:,0].type(torch.FloatTensor).mean())
    STAT_vector = np.zeros(N_per)
    for r in range(N_per):
        ind = np.random.choice(N, N, replace=False)
        # divide into new X, Y
        ind_X = ind[:N1]
        ind_Y = ind[N1:]
        # print(indx)
        STAT_vector[r] = abs(output[ind_X,0].type(torch.FloatTensor).mean() - output[ind_Y,0].type(torch.FloatTensor).mean())
    S_vector = np.sort(STAT_vector)
    threshold = S_vector[np.int(np.ceil(N_per * (1 - alpha)))]
    h = 0
    if STAT.item() > threshold:
        h = 1
    return h, threshold, STAT
