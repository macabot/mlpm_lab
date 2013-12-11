import scipy.special as sp
from matplotlib import pyplot as plt
import numpy as np

class BayesianPCA(object):

    def __init__(self, d, N, a_alpha=10e-3, b_alpha=10e-3, a_tau=10e-3, b_tau=10e-3, beta=10e-3):
        """
        Initializes parameters randomly such that they can be used
        by updating later
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

        # set data 
        Sigma = np.diag([5,4,3,2,1,1,1,1,1,1])
        self.data = np.random.multivariate_normal(np.zeros(d), Sigma, N).T


    def __update_z(self, X):
        """
        updates X 
        X is the prod over datapoints: N(x_n | m_x, Sigma_n)
        X[0][n] stands for nth mean
        X[1] stands for sigma

        m_x = <tau> * E_x * <W^t> * (t_n - <mu>)
        E_x = (I + <tau> <W^t*W>)^-1

        # TODO: what is self.means_z and self.sigma_z if not those of X?
        """

        # variables necessary in calculations
        tau_exp = self.a_tau_tilde / self.b_tau_tilde
        mu_exp = self.mean_mu
        WT_W_exp = 0; # TODO
        WT_exp = self.means_w.T

        # update sigma, equal for all n's
        X[1] = np.linalg.inv(np.eye(self.d,self.d) + tau_exp * WT_W_exp)

        # update mean TODO
        #X[0] = np.dot(np.dot( tau_exp * X[1], WT_exp), (self.data - mu_exp))
        X[0] = np.random.randn(self.data.shape[0], self.data.shape[1])

    def __update_mu(self, X):
        """
        update mean_mu and sigma_mu

        sigma_mu = (beta + N * <tau>)^-1 * I
        mean_mu = <tau> * sigma_mu * sum( data_n - <W> * <X_n> ) 
        """

        # necessary for calculations of both
        tau_exp = self.a_tau_tilde / self.b_tau_tilde
       
        # update sigma_mu first as it is used in updating mean_mu
        self.sigma_mu = (1.0 / (self.beta + self.N * tau_exp)) * np.eye(self.d)

        # update mean_mu
        m_x, sigma_x = X
        sum_t_w_x = np.sum(self.data - np.dot(self.means_w, m_x), axis=1, keepdims=True)
        self.mean_mu = np.dot(tau_exp * self.sigma_mu, sum_t_w_x)


    def __update_w(self, X):
        """
        update mean_w and sigma_w

        sigma_w ( diag<alpha> + <tau> * sum( <x_n * x_n.T  )  )
        mean_w_k = <tau> * sigma_w * sum( <x_n> * ( t_nk - <mu_k>  )  )
        """

        # necessary for calculations
        tau_exp = self.a_tau_tilde / self.b_tau_tilde


        # update sigma_w first as it is used in updating means_w
        m_x, sigma_x = X
        # TODO: is the expected value over alpha not a mixture?
        diag_exp_alpha = np.diag(self.a_alpha_tilde / self.b_alpha_tilde)
        tau_sum_xn = tau_exp * np.sum(np.dot(m_x, m_x.T), axis=1) 
        self.sigma_w = np.linalg.inv(diag_exp_alpha + tau_sum_xn)

        # update means_w
        # einsum calculates for all k the summation over <x_n> * ( t_nk - mu_k)
        einsum_result = np.einsum('kj,ij->ik', self.data - self.mean_mu, X[0])
        self.means_w = np.dot(tau_exp * self.sigma_w, einsum_result)

    def __update_alpha(self):
        """ 
        update b_alpha_tilde, a_alpha_tilde does not change 

        b_alpha_tilde[i] = b_alpha + < || w[i] ||^2 > / 2
        """

        # update each element in b_alpha_tilde 
        # TODO: not sure, correction: pretty sure it is wrong
        w_norm = np.power(np.linalg.norm(self.means_w, axis=1),2)
       
        self.b_alpha_tilde = self.b_alpha + w_norm / 2

        exit()

    def __update_tau(self, X):
        t_norm_sq = np.power(np.linalg.norm(self.data))
        mu_norm_sq = np.power(np.linalg.norm(self.mean_mu))
        t_mu = t_norm_sq + mu_norm_sq
        # TODO continue

    def L(self, X):
        L = 0.0
        return L

    def fit(self, X):

        # set constant parameters
        self.a_alpha_tilde = self.a_alpha + self.d / 2
        self.a_tau_tilde = self.a_tau + self.N*self.d / 2

        it = 1000
        converged = False
        while not converged and it > 0: 
            print("Updating iteration " + str(it))
            # run each update
            self.__update_z(X)
            self.__update_mu(X)
            self.__update_w(X)
            self.__update_alpha()
            self.__update_tau(X)

            it -= 1
            # TODO: decide on converged


def run():
    vpca = BayesianPCA(10, 2)
    X = [[0],0] # set X random
    vpca.fit(X)

if __name__ == '__main__':
    run()

