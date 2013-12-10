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
        self.b_alpha_tilde = np.abs(np.random.randn(d, 1))
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
        W_exp = np.multiply.reduce(self.means_w)
        mu_exp = self.mean_mu

        # update sigma, equal for all n's
        X[1] = np.linalg.inv(np.eye(d,d) + np.dot(tau_exp, np.dot(W_exp.T, W_exp)))

        # update mean
        X[0] = np.dot(np.dot( tau_exp * X[1], W_exp.T), (self.data - mu_exp))


    def __update_mu(self, X):
        """update mean_mu and sigma_mu"""
        m_x, sigma_x = X
        tau_exp = self.a_tau_tilde / self.b_tau_tilde

        sum_t_w_x = np.sum(self.data - np.dot(self.means_w, m_x), axis=1)
        self.mean_mu = np.dot(np.dot(tau_exp, self.sigma_mu), sum_t_w_x)

        beta_n_tau-inv = 1.0 / (self.beta + self.N * tau_exp)
        self.sigma_mu = beta_n_tau_inv * np.eye(self.d) # TODO check size of I

    def __update_w(self, X):
        """update mean_w and sigma_w"""
        m_x, sigma_x = X
        tau_exp = self.a_tau_tilde / self.b_tau_tilde
        sum_x_t_mu = np.sum(np.dot(m_x, (self.data - self.mean_mu)), axis=1)
        self.mean_w = np.dot(np.dot(tau_exp, self.sigma_w), sum_x_t_mu)

        diag_exp_alpha = np.diag(self.a_alpha_tilde / self.bs_alpha_tilde)
        tau_sum_xn = tau_exp * np.sum(np.dot(m_x, m_x), axis=1)
        self.sigma_w = np.linalg.inv(diag_exp_alpha + tau_sum_xn)

    def __update_alpha(self):
        """ 
        update b_alpha_tilde, a_alpha_tilde does not change 

        b_alpha_tilde[i] = b_alpha + < || w[i] ||^2 > / 2
        """

        # variables necessary in calculations
        # todo: not sure if to power 2
        w_norm = np.power(np.linalg.norm(self.means_w, axis=1),2)
        
        # update each element in b_alpha_tilde (todo: remove for loop)
        self.b_alpha_tilde = self.b_alpha + w_norm / 2

    def __update_tau(self, X):
        t_norm_sq = np.power(np.linalg.norm(self.data))
        mu_norm_sq = np.power(np.linalg.norm(self.mean_mu))
        t_mu = t_norm_sq + mu_norm_sq
        # TODO continue

    def L(self, X):
        L = 0.0
        return L

    def fit(self, X):

        # todo: fit a_alpha

        pass
