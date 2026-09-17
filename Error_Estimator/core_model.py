from gpytorch.utils.quadrature import GaussHermiteQuadrature1D
from torch.distributions import MultivariateNormal
from utils import *
from Error_Estimator.core_gp_model import CoreGPModel
import torch.nn.functional as F
import gpytorch
import torch
import torch.nn as nn
from utils import psd_safe_cholesky




def _triangular_inverse(A, upper=False):
    eye = torch.eye(A.size(-1), dtype=A.dtype, device=A.device)
    return eye.triangular_solve(A, upper=upper).solution


class VariationalELBO(gpytorch.Module):

    def __init__(self, model, num_data, num_inducing_points, lr=0.1, dtype=torch.float64):
        super().__init__()
        self.model = model
        # local parameters
        
        self.lr = lr
        self.num_data = num_data

        # global parameters
        self.num_inducing_points = num_inducing_points
        scaled_mean_init = torch.zeros(self.num_inducing_points, dtype=dtype)
        neg_prec_init = torch.eye(self.num_inducing_points, self.num_inducing_points, dtype=dtype).mul(-0.5)
        # eta and H parameterization of the variational distribution

        self.total_tokens = 1#total_tokens
        self.output_dim = 294912#output_dim
        self.num_inducing_points = num_inducing_points
        self.register_buffer(
                      "eta",
                      torch.zeros(
                      self.total_tokens,
                      self.output_dim,
                      self.num_inducing_points,
                      dtype=dtype))
        self.D = 294912#168960#294912
        self.N = 58
        self.B = 1
        self.M = num_inducing_points

        self.register_buffer(
             "H", -0.5 * torch.eye(
             self.num_inducing_points,
             dtype=dtype
             ).unsqueeze(0).repeat(
              self.total_tokens,
              1, 1) )
        #self.register_buffer("noise",torch.tensor(1e-4, dtype=dtype))
        self.noise = torch.tensor(1e-4, dtype=torch.float32)
        
    def forward(self, K, Y, batch_idx):
        
       
        
        M = self.num_inducing_points

      
        if Y.ndim == 2:
            Y = Y.unsqueeze(1)

        mu, Sigma = self.NaturalToMuSigma(batch_idx)
        mu = mu.to(torch.float32)
        Sigma = Sigma.to(torch.float32)
   

        Kmm = K[:M, :M]
        Knm = K[:M, M:].t()
        Knn = K[M:, M:]

  

        L = psd_safe_cholesky(Kmm)



        kappa = torch.cholesky_solve(
         Knm.t(),
         L
        ).t()



        kappaMu = torch.einsum(
          "nm,bdm->bdn",
           kappa,
            mu
          )

        mu_f = kappaMu.permute(0, 2, 1)



        K_tilde = torch.diagonal(
        Knn - Knm @ kappa.t()
             )



        kappaSigmakappa = torch.einsum(
          "nm,bmk,nk->bn",
           kappa,
           Sigma,
           kappa
          )



        predictive_var = (
          K_tilde.unsqueeze(0)
          + kappaSigmakappa
       )


        predictive_var = predictive_var.unsqueeze(-1).expand( self.B ,self.N, self.D)#(B, N, D)



        Y_bnd = Y.permute(1, 0, 2)


        expected_sq_error = (
         (Y_bnd - mu_f).pow(2)
         + predictive_var
        )

        expected_log_likelihood = -0.5 * (expected_sq_error / self.noise + torch.log(2.0 * torch.pi * self.noise)).sum()



        expected_log_likelihood = (
           expected_log_likelihood.sum()
      )



        sign_K, logdet_Kmm = torch.linalg.slogdet(Kmm)

        if torch.any(sign_K <= 0):
           raise RuntimeError(
            "Kmm is not positive definite."
           )

        sign_S, logdet_Sigma = torch.linalg.slogdet(Sigma)

        if torch.any(sign_S <= 0):
            raise RuntimeError(
            "Variational Sigma is not positive definite."
          )



        Kmm_inv_Sigma = torch.cholesky_solve(
            Sigma,
            L
         )

        trace_term = torch.diagonal(
           Kmm_inv_Sigma,
           dim1=-2,
           dim2=-1
          ).sum(dim=-1)


        Kmm_inv_mu = torch.cholesky_solve(
            mu.reshape(1 * self.D, self.M).t(),
            L
             ).t().reshape(1, self.D, self.M)

        quad_term = (
          mu * Kmm_inv_mu
         ).sum(dim=-1)


        KL_per_output = 0.5 * (
          trace_term.unsqueeze(-1)
          + quad_term
          - M
          + logdet_Kmm
          - logdet_Sigma.unsqueeze(-1)
          )



        KL = KL_per_output.sum()


        ELBO = (
             expected_log_likelihood
            - KL
          )


        self.ctx = {
         "Kmm": Kmm.detach(),
         "kappa": kappa.detach(),
         "Y": Y.detach(),
         "batch_idx": batch_idx.detach(),
         "predictive_var": predictive_var.detach(),
        }

        return ELBO

    def NaturalToMuSigma(self, batch_idx):

       eta = self.eta[batch_idx]
       H = self.H[batch_idx]

       

       precision = -2.0 * H


       L_precision = psd_safe_cholesky(precision)


       M = H.shape[-1]

       I = torch.eye(
        M,
        dtype=H.dtype,
        device=H.device).expand(H.shape[0], -1, -1)

       Sigma = torch.cholesky_solve(
        I,
        L_precision
       )


       mu = torch.einsum(
        "bij,bdj->bdi",
        Sigma,
        eta
       )


       return mu, Sigma

    def update(self):
        
       

       Kmm = self.ctx["Kmm"]
       kappa = self.ctx["kappa"]
       Y = self.ctx["Y"]
       batch_idx = self.ctx["batch_idx"]

       with torch.no_grad():



         N_batch = Y.shape[0]
         B = Y.shape[1]
         D = Y.shape[2]

         M = self.num_inducing_points



         data_scale = self.num_data / N_batch


         L = psd_safe_cholesky(Kmm)

         I = torch.eye(
            M,
            dtype=Kmm.dtype,
            device=Kmm.device
         )

         Kmm_inv = torch.cholesky_solve(
            I,
            L
         )


         Y_bdn = Y.permute(1, 2, 0)

         eta_target = (
            data_scale / self.noise
            * torch.einsum(
                "nm,bdn->bdm",
                kappa,
                Y_bdn
            )
         )


         kappa_t_kappa = kappa.t() @ kappa

         H_target = -0.5 * (
            Kmm_inv
            +
            data_scale / self.noise
            * kappa_t_kappa
         )


         self.eta[batch_idx] += (
            self.lr
            * (
                eta_target
                - self.eta[batch_idx]
            )
         )

         self.H[batch_idx] += (
            self.lr
            * (
                H_target.unsqueeze(0)
                - self.H[batch_idx]
            )
         )

class Core_Model(gpytorch.Module):
    def __init__(self,
                 kernel_func,
                 dtype=torch.float32,
                 num_inducing_points=2,  # number of inducing points to use
                 lengthscale=1.,
                 num_data=100,
                 natural_lr=0.1):

        super(Core_Model, self).__init__()
        
        self.num_inducing_points = num_inducing_points  # number of inducing points per class
        self.kernel_func = kernel_func

        self.dtype = dtype
        self.loss_fn = nn.CrossEntropyLoss()

        self.model = CoreGPModel(kernel_func, jitter_val=1e-2)
        self.ELBO = VariationalELBO(self.model, num_data, num_inducing_points, natural_lr, dtype)
        
        self.num_token = [0]#24
        self.token_idx = torch.tensor(self.num_token,dtype=torch.long)
        
    def forward_mll(self, Points, Y, batch_idx):

        Y_n = Y#.mul(2).sub(1).to(self.dtype)  # [0, 1] -> [-1, 1]
        _, K = self.model(Points)
        #print('KKKKKKKKKKKK', K.size())
        
        mll = self.ELBO(K, Y_n, self.token_idx)
        return mll

    @torch.no_grad()
    def predictive_posterior(
        self,
        Points,
        jitter_Kmm=False,
        include_noise=False
      ): 
        

        _, K = self.model(Points)
        M = self.num_inducing_points

        I = torch.eye(
          M,
          dtype=K.dtype,
          device=K.device
        )


        
        mu, Sigma = self.ELBO.NaturalToMuSigma(
          self.token_idx
         )


        B = mu.shape[0]
        D = mu.shape[1]


        Kmm = K[:M, :M]

        if jitter_Kmm:
          Kmm = Kmm + 0.03 * I

        Knm = K[:M, M:].t()

        Knn = K[M:, M:]


        L = psd_safe_cholesky(Kmm)


        kappa = torch.cholesky_solve(
           Knm.t(),
           L
        ).t()


        mu = mu.to(torch.float32)
        mu_s = torch.einsum(
         "nm,bdm->bnd",
          kappa,
          mu
          )


        K_tilde = torch.diagonal(
           Knn - Knm @ kappa.t()
         )


        Sigma = Sigma.to(torch.float32)
        kappaSigmakappa = torch.einsum(
          "nm,bmk,nk->bn",
          kappa,
          Sigma,
          kappa
         )



        Sigma_s = (
          K_tilde.unsqueeze(0)
          +
          kappaSigmakappa
          )



        Sigma_s = Sigma_s.unsqueeze(-1).expand(
          B,
          Sigma_s.shape[1],
          D
        )


        if include_noise:
          Sigma_s = Sigma_s + self.noise

        return mu_s, Sigma_s