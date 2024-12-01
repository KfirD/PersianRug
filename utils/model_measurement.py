import numpy as np
import torch as t
import os 
from . import model_analytics as ma
import dill
from dataclasses import dataclass
import hadamard_model as hm

class ModelMeasurement():
    def __init__(self, model) -> None:
        self.cfg = model.cfg
        self.p_feat = model.p_feat

        self.final_loss = model.final_loss()
        W = model.W_matrix().detach().clone()
        self.diag = t.diag(W)
        self.diag_var = self.diag.var().item()
        self.diag_mean = self.diag.mean().item()
        self.chi_statistic, self.chi_pval = ma.get_model_gaussian_ks(model, row=0)

        self.chi_meanvar = ma.chi_mean_variance_over_rows(model)
        self.chi_varvar = ma.chi_varvar(model)
        self.bias_var = model.final_layer.bias.detach().var().item()

        eps_mat = ma.get_eps_mat(W)
        # self.eps_mean = eps_mat.sum(dim=1)        
        # self.eps_mean_var = self.eps_mean.var(dim=0)

        self.abseps3 = ma.get_abseps3(eps_mat)
        self.eps2 = ma.get_eps2(eps_mat)
        self.Lyapunov = ma.get_Lyapunov_min_max_avg(eps_mat)
        
        # self.eigs = np.linalg.eigvalsh(W)
        self.anti_symmetric_norm = t.norm(0.5*(W - W.T)).item()
        self.symmetric_norm = t.norm(0.5*(W + W.T)).item()
        self.W_norm = t.norm(W).item()
        # self.rip_constant = ma.estimate_rip_constant(model.initial_layer.weight.data.detach().cpu(), self.cfg.n_dense)
        # self.Win_mutual_coherence = ma.mutual_coherence(model)
        # self.model_id = model_id
        # self.losses = model.losses

    def ratio(self):
        return self.cfg.n_dense/self.cfg.n_sparse
    
    def final_loss(self):
        return self.final_loss
    
    def save(self, filename):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)        
        with open(filename, 'wb') as f:
            dill.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            measurement = dill.load(f)
        
        return measurement
    

class HadamardModelMeasurement():
    def __init__(self, model:hm.HadamardModel) -> None:
        self.cfg = model.cfg
        self.p_feat = model.p_feat

        self.final_loss = model.final_loss()
        W = model.W_matrix().detach().clone()
        self.diag = t.diag(W)
        self.diag_var = self.diag.var().item()
        self.diag_mean = self.diag.mean().item()
        self.chi_statistic, self.chi_pval = ma.get_model_gaussian_ks(model, row=0)

        # self.chi_meanvar = ma.chi_mean_variance_over_rows(model)
        # self.chi_varvar = ma.chi_varvar(model)
        # self.bias_var = model.final_layer.bias.detach().var().item()

        eps_mat = ma.get_eps_mat(W)

        self.abseps3 = ma.get_abseps3(eps_mat)
        self.eps2 = ma.get_eps2(eps_mat)
        self.Lyapunov = ma.get_Lyapunov_min_max_avg(eps_mat)
        
        # self.eigs = np.linalg.eigvalsh(W)
        self.anti_symmetric_norm = t.norm(0.5*(W - W.T)).item()
        self.symmetric_norm = t.norm(0.5*(W + W.T)).item()
        self.W_norm = t.norm(W).item()
        # self.rip_constant = ma.estimate_rip_constant(model.initial_layer.weight.data.detach().cpu(), self.cfg.n_dense)
        # self.Win_mutual_coherence = ma.mutual_coherence(model)
        # self.model_id = model_id
        self.losses = model.losses

    def ratio(self):
        return self.cfg.n_dense/self.cfg.n_sparse
    
    def save(self, filename):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)        
        with open(filename, 'wb') as f:
            dill.dump(self, f)
    
    @classmethod
    def load(cls, filename):
        with open(filename, 'rb') as f:
            measurement = dill.load(f)
        
        return measurement
            
