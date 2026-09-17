from Error_Estimator.core_model import Core_Model
#from error_learner.shared_SVGP import Core_Model
from torch import nn
import torch.optim as optim
from utils import *
import gpytorch
import torch

class Runner(gpytorch.Module):
    def __init__(self, args, device):
        super(Runner, self).__init__()
        self.model = None
        self.device = None
        self.args = args
        self.id = 0
        self.depth = 0
        self.num_data = 1

        self.X_support = None
        self.Y_support = None
        self.state = None




class RUN_VI(Runner):



    def set_model(self, kernel_function, num_inducing_points, lengthscale, natural_lr,
                  outputscale, dtype, num_data):

        
        self.num_data = num_data
        
        

        dtype=torch.float32
        # number of inducing points is at most the number of data points
        self.model = Core_Model(kernel_func=kernel_function, dtype=dtype,
                           num_inducing_points=num_inducing_points, lengthscale=lengthscale, num_data=num_data, natural_lr=natural_lr)

        self.model.model._set_params(outputscale=outputscale, lengthscale=lengthscale)

        self.model.to(self.device)
        

    def train_core(self, X, Y, batch_idx, Z, tk_idx):
        #print('@@@@@@@@@@', Z.size(), X.size())
        train_data = torch.cat((Z, X), dim=0)
        #dist2 = ((Z[:, None, :] - X[None, :, :]) ** 2).sum(dim=-1)

        #print("dist2 min:", dist2.min().item())
        #print("dist2 max:", dist2.max().item())
        #print("dist2 mean:", dist2.mean().item())
        loss = - self.model.forward_mll(train_data, Y, batch_idx)

        avg_loss = loss.item() / self.num_data
        
        #if self.args.train_strategy == 'hierarchical':
        #  self.model.ELBO.update()
        #elif tk_idx == 95:
        self.model.ELBO.update()  # update natural parameters.
        #print("AFTER UPDATE eta:",self.model.ELBO.eta[tk_idx].abs().max().item())
        return loss / self.num_data
        #return loss.detach() / self.num_data

    def eval_core(self, X, Z):
        #print(Z.size(), X.size())
        X_star = torch.cat((Z, X), dim=0)
        #print("dist2 min:", dist2.min().item())
        #print("dist2 max:", dist2.max().item())
        #print("dist2 mean:", dist2.mean().item())
        #print("X_star", X_star.size())
        mu, sigma = self.model.predictive_posterior(X_star)

        return mu, sigma

