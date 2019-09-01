import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import Parameter
import numpy as np

def log_gaussian_mixture(w, *args):

    pi, sigma_1, sigma_2 = args
    w_stacked = torch.stack(
            [-w**2/(2*sigma_1**2) + torch.log(pi) - 0.5*torch.log(2*np.pi*sigma_1**2),
            -w**2/(2*sigma_2**2) + torch.log(1 - pi) - 0.5*torch.log(2*np.pi*sigma_2**2)],
            dim = -1
            )
    log_P_w = torch.logsumexp(w_stacked, dim = -1)
    return log_P_w

class Gaussian():
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

    def log_prob(self, x):
        #const. 1e-8 added for numerical stability
        return -0.5*torch.log(2*np.pi*(self.scale**2) + 1e-8) - (x - self.loc)**2/(2*self.scale**2 + 1e-8)

class Bayesian_Linear(nn.Module):
    def __init__(self, in_features, out_features, pi = 0.5, sigma1 = 2, sigma2 = 0.5,
            fill_mu = 0, fill_rho = -5, name = None, bias=True, local_repar = False):

        super(Bayesian_Linear, self).__init__()

        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'

        self.in_features = in_features
        self.out_features = out_features
        self.local_repar = local_repar
        self.name = name
        self.mixture_args = torch.Tensor([pi, sigma1, sigma2]).to(device)
        self.fill_mu = fill_mu; self.fill_rho = fill_rho

        self.mu_W = Parameter(torch.Tensor(out_features, in_features))
        self.rho_W = Parameter(torch.Tensor(out_features, in_features))
        self.mu_b = Parameter(torch.Tensor(1, out_features))
        self.rho_b = Parameter(torch.Tensor(1, out_features))

        self.normal = Gaussian(torch.Tensor([0.0]).to(device), torch.Tensor([1.0]).to(device))

        self.reset_parameters()

    def reset_parameters(self):
        self.mu_W.data.fill_(self.fill_mu)
        self.rho_W.data.fill_(self.fill_rho)
        self.mu_b.data.fill_(self.fill_mu)
        self.rho_b.data.fill_(self.fill_rho)

    def forward(self, x):
        self.eps_W = self.mu_W.data.new(self.mu_W.size()).normal_()
        self.eps_b = self.mu_b.data.new(self.mu_b.size()).normal_()
        self.sigma_W = torch.log(1 + torch.exp(self.rho_W))
        self.sigma_b = torch.log(1 + torch.exp(self.rho_b))
        self.W = self.mu_W + torch.log(1 + torch.exp(self.rho_W))*self.eps_W
        self.b = self.mu_b + torch.log(1 + torch.exp(self.rho_b))*self.eps_b

        return F.linear(x, self.W) + self.b

    def KL(self):
        log_q_W = (self.normal.log_prob(self.eps_W) - torch.log(self.sigma_W + 1e-8)).sum()
        log_q_b = (self.normal.log_prob(self.eps_b) - torch.log(self.sigma_b + 1e-8)).sum()
        log_P_W = log_gaussian_mixture(self.W, *self.mixture_args).sum()
        log_P_b = log_gaussian_mixture(self.b, *self.mixture_args).sum()

        self.KL_loss = log_q_W + log_q_b - log_P_W - log_P_b
        return self.KL_loss

class Bayesian_Net(nn.Module):
    def __init__(self):
        super(Bayesian_Net, self).__init__()
        self.fc1 = Bayesian_Linear(28*28, 300,)
        self.fc2 = Bayesian_Linear(300,  300)
        self.fc3 = Bayesian_Linear(300,  10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x

class ELBO(nn.Module):
    def __init__(self, net, data_size, task):
        super(ELBO, self).__init__()
        self.net = net
        self.data_size = data_size
        self.task = task

    def forward(self, input, target):
        self.KL_loss = 0
        if self.task == 'regression':
            self.data_distr = Gaussian(input[:,0],input[:,1])
            self.neg_log_data_likelihood = self.data_distr.log_prob(target).mean()
        else:
            self.neg_log_data_likelihood = F.cross_entropy(input, target)

        for layer in self.net.children():
            if hasattr(layer, 'KL'):
                self.KL_loss += layer.KL()
        self.KL_loss /= self.data_size
        ELBO = self.KL_loss + self.neg_log_data_likelihood
        return ELBO
