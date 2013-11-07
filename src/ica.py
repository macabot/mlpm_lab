from numpy import *
from matplotlib.pyplot import *


def sawtooth(x, period=0.2, amp=1.0, phase=0.):
    return (((x / period - phase - 0.5) % 1) - 0.5) * 2 * amp

def sine_wave(x, period=0.2, amp=1.0, phase=0.):
    return np.sin((x / period - phase) * 2 * np.pi) * amp

def square_wave(x, period=0.2, amp=1.0, phase=0.):
    return ((np.floor(2 * x / period - 2 * phase - 1) % 2 == 0).astype(float) - 0.5) * 2 * amp

def triangle_wave(x, period=0.2, amp=1.0, phase=0.):
    return (sawtooth(x, period, 1., phase) * square_wave(x, period, 1., phase) + 0.5) * 2 * amp

def random_nonsingular_matrix(d=2):
    """
    Generates a random nonsingular (invertible) matrix if shape d*d
    """
    epsilon = 0.1
    A = np.random.rand(d, d)
    while abs(np.linalg.det(A)) < epsilon:
        A = np.random.rand(d, d)
    return A

def plot_signals(X):
    """
    Plot the signals contained in the rows of X.
    """
    figure()
    for i in range(X.shape[0]):
        ax = subplot(X.shape[0], 1, i + 1)
        plot(X[i, :])
        ax.set_xticks([])
        ax.set_yticks([])

def make_mixtures(S, A):
    """ (matrix, matrix) -> matrix
    Returns the mixure of two matrixes

    Assumes X and A are of type numpy matrix
    """
                
    return A*S

def plot_histograms(X):
    """
    Plot the signals contained in the rows of X as a histogram
    """
    figure()
    for i in range(X.shape[0]):
        ax = subplot(X.shape[0], 1, i + 1)
        hist(X[i, :], bins=20)
        ax.set_xticks([])
        ax.set_yticks([])

def generate_data():
    """ Generating data  """
    num_sources = 6
    signal_length = 500
    t = np.linspace(0, 1, signal_length)
    S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size), np.random.rand(t.size)].T
    plot_signals(S)
    plot_histograms(S)

    show()

def whiten(data):
    """whitening"""
    covariance = np.cov(data)
    # The columns of phi are the eigenvectors of the covariance matrix.
    #eigenvalues, eigenvectors = np.linalg.eig(covariance)
    #print 'val: %s' % eigenvalues
    #print 'vec: %s' % eigenvectors
    phi = np.linalg.eig(covariance)[1]
    diag_lambda = np.dot(np.dot(phi.T, covariance), phi)
    diag_lambda = np.diag(np.diag(diag_lambda))
    print diag_lambda
    
def test_whitening():
    #data = np.matrix([[1,1], [2,2], [3,0]]).T
    data = np.random.randn(2, 1000000)*1000
    whiten(data)
    
if __name__ == '__main__':
    test_whitening()
