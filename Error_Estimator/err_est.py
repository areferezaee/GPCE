import torch.nn as nn
from utils import *
from gpytorch.models import ApproximateGP, GP
import gpytorch
from Error_Estimator.runner import RUN_VI
import torch

class Model(gpytorch.Module):
    def __init__(self, args):
        super(Model, self).__init__()
        self.args = args
        self.device = torch.device('cpu')#('cuda')
        self.criterion = nn.CrossEntropyLoss()
        self.RUN_VI = RUN_VI(args, self.device)

    def _init_Xbar(self, X, Y):
        raise NotImplementedError("not yet implemented")

    def forward(self, x, y, x_idx, mode='train', tk_idx=0):
        raise NotImplementedError("not yet implemented")


  

class ModelErrorEstimator(Model):
    def __init__(self, args, device, pretrained=True):
        super(ModelErrorEstimator, self).__init__(args)
        self.learn_location = 'True'

        Xbar_dim = (self.args.num_inducing_points, self.args.feature_length[0])#[-1])#, self.args.feature_length[1])
        if self.learn_location:
            self.Xbar = nn.Parameter(torch.randn(Xbar_dim), requires_grad=True)
        else:
            self.Xbar = torch.randn(Xbar_dim).to(device)
        
    def _init_Xbar(self, X, Y):
        
          with torch.no_grad():
            
            num_inducing = self.args.num_inducing_points
            
            Xbar = self.Xbar 
            self.Xbar = Xbar
    
    def forward(self, x, y, x_idx, mode='train', tk_idx=0):       
      if mode == 'train':
        z = x
        lengthscale = 1.
        num_inducing_inputs = self.args.num_inducing_points#self.Xbar.shape[1] * self.Xbar.shape[2]
        #print('self.Xbar.dtype@@@@@@@@@@', self.Xbar.dtype)
        self.RUN_VI.set_model(kernel_function=self.args.kernel_function, num_inducing_points=num_inducing_inputs, lengthscale=lengthscale,
                               natural_lr=self.args.natural_lr, outputscale=self.args.outputscale,
                               dtype=self.Xbar.dtype, num_data=self.args.feature_length[-1])

        
        
        loss = self.RUN_VI.train_core(z, y, x_idx, self.Xbar,tk_idx=0)
        return loss

      else:  
        z = x
            
        with torch.no_grad():
             mu, sigma = self.RUN_VI.eval_core(z, self.Xbar)
             mse = torch.mean((y - mu) ** 2)
             mae = torch.mean(torch.abs(y - mu))
             eps = 1e-8

             nlpd = 0.5 * ( torch.log(2 * torch.pi * sigma + eps) + (y - mu) ** 2 / (sigma + eps))
             nlpd = nlpd.mean()
        #loss = self.criterion = (y, preds)
        
        return mse, mae, nlpd, mu





  

    
