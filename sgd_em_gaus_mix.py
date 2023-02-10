import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class SGDEMGauss(nn.Module):
    
    def __init__(self, n_dim: int, n_clusters: int = 4, init_par: dict = {}):
        super().__init__()
        self.n_dim = n_dim
        self.n_clusters = n_clusters
        self.tril_idx = torch.tril_indices(n_dim, n_dim)
        if 'mu' not in init_par:
            self.mu = torch.randn(self.n_clusters, self.n_dim, requires_grad = True)
        else:
            self.mu = init_par['mu']
            self.mu.requires_grad_(True)
        if 'sigma' not in init_par:
            #Cholesky parametrization of covariance-variance matrices
            self.low_par = torch.rand(self.n_clusters, 
                                      int(self.n_dim * (self.n_dim + 1) / 2 ), 
                                      requires_grad = True)
        else:
            #We need a tensor with shape (n_clusters, n_dim, n_dim)
            if init_par['sigma'].shape != (self.n_clusters,
                                           self.n_dim,
                                           self.n_dim):
                raise Exception(
                    "Shape of init_par['sigma'] must be (n_clusters, n_dim, n_dim)")
                lower = torch.linalg.cholesky(init_par['sigma'])
                self.low_par = lower[:, self.tril_idx[0, :], self.tril_idx[1, :]]
                self.low_par.requires_grad_(True)
        
        #Register parameters
        self.mu = nn.Parameter(self.mu)
        self.low_par = nn.Parameter(self.low_par)
    
    def lower_tri(self):
        lower = torch.zeros(self.n_clusters, self.n_dim, self.n_dim)
        lower[:, self.tril_idx[0, :], self.tril_idx[1, :]] = self.low_par[:, :]
        return lower
    
n_clusters = 2
n_dim = 3
sgd_gauss = SGDEMGauss(n_dim, n_clusters, {})

b_size = 4
low1 = 2 * torch.eye(n_dim, requires_grad = True)
low2 = torch.eye(n_dim, requires_grad = True)
low = torch.concat((low1, low2), dim = 0).reshape(n_clusters, n_dim, n_dim)
low_t = low.T
mu = torch.tensor( [[1.0]*n_dim, [2.0]*n_dim], requires_grad=True)
x = torch.rand(b_size, n_dim)
sigma_inv = torch.zeros(n_clusters, n_dim, n_dim)

for k in range(n_clusters):
    sigma_inv[k, :, :] = low[k, :, :] @ low_t[:, :, k]
sigma_inv = torch.linalg.inv(sigma_inv)

mu = mu.view(n_clusters, 1, n_dim)
x = x.view(1, b_size, n_dim)
#diff has shape (n_clust, batch_size, n_dim)
diff = x - mu
sigma_inv = sigma_inv.view(n_clusters, 1, n_dim, n_dim)
diff = diff.view(n_clusters, b_size, n_dim, 1)
prod1 = sigma_inv @ diff
prod1 = prod1.squeeze(dim = -1)
diff = diff.squeeze(dim = -1)
prod2 = (diff * prod1).sum(axis = -1)

#Validation
idx_clust = 1
idx_b = 1
l = low[idx_clust, :, :]
m = mu[idx_clust, :]
s_inv = torch.linalg.inv(l @ l.T)
obs = x[0, idx_b, :]
d = (obs-m).view(n_dim, 1)
res = s_inv @ d
res = d.T @ res
print(res)
print(prod2[idx_clust, idx_b])


class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.mean = nn.Parameter(torch.zeros(1))
        self.pdf = torch.distributions.Normal(self.mean, torch.tensor([1.0]))

    def forward(self, x):
        return -self.pdf.log_prob(x)
    
model = GaussianModel()

from pandas_datareader.yahoo.daily import YahooDailyReader
import pandas as pd

symbols = pd.read_csv('./data/sp500_symbols.csv', index_col = "Symbol")
symbols = [s[0] for s in symbols.values]
reader = YahooDailyReader(symbols = symbols,
                         start = "01/01/15",
                         end = "12/31/22")
data = reader.read()
