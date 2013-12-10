import scipy.special as sp
from matplotlib import pyplot as plt
import numpy as np

class BayesianPCA(object):
    
    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        """
        self.d = d # number of dimensions
        self.N = N # number of data points
        
        # Hyperparameters
        self.a_alpha = a_alpha
        self.b_alpha = b_alpha
        self.a_tau = a_tau
        self.b_tau = b_tau
        self.beta = beta

        # Variational parameters
        self.means_z = np.random.randn(d, N) # called x in bishop99
        self.sigma_z = np.random.randn(d, d)
        self.mean_mu = np.random.randn(d, 1)
        self.sigma_mu = np.random.randn(d, d)
        self.means_w = np.random.randn(d, d)
        self.sigma_w = np.random.randn(d, d)
        self.a_alpha_tilde = np.abs(np.random.randn(1))
        self.bs_alpha_tilde = np.abs(np.random.randn(d, 1))
        self.a_tau_tilde = np.abs(np.random.randn(1))
        self.b_tau_tilde = np.abs(np.random.randn(1))
    
        # set data TODO
        self.data = 0

    def __update_z(self, X):
        """ 
        updates X (projection of data points?)
        X is the prod over datapoints: N(x_n | m_x, Sigma_n)
        X[0][n] stands for nth mean
        X[1] stands for sigma

        m_x = <tau> * E_x * <W^t> * (t_n - <mu>)
        E_x = (I + <tau> <W^t*W>)^-1
        """
        
        # variables necessary in calculations
        tau_exp = self.a_tau_tilde / self.b_tau_tilde
        W_exp_T = np.multiply.reduce(self.means_w)
        mu_exp = self.mean_mu

        # update sigma, equal for all n's
        X[1] = (np.ones(

        # update mean
        X[0] = np.dot(np.dot(np.dot(tau_exp, X[1]), W_exp_T), (self.data - mu_exp))
    
    def __update_mu(self):
   
    def __update_w(self, X):
        pass
    
    def __update_alpha(self):
        pass

    def __update_tau(self, X):
        pass

    def L(self, X):
        L = 0.0
        return L
    
    def fit(self, X):
        pass
