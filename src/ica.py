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

    return dot(A, S)

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
    mean = np.mean(data)
    data -= mean
    covariance = np.cov(data)
    # The columns of phi are the eigenvectors of the covariance matrix.
    phi = np.linalg.eig(covariance)[1]
    diag_lambda = np.diag(np.dot(np.dot(phi.T, covariance), phi))
    return np.dot(np.dot(np.diag(diag_lambda**-0.5), phi.T), data)
    
def plot_functions():
    figure()
    ranges = range(0,100)
    print ranges
    ranges = np.linspace(-1,1,100)
    print ranges
    outputs = np.c_[[1/cosh(x) for x in ranges],
                    [-tanh(x) for x in ranges],
               [exp(((-x**2)/2) + np.log(cosh(x)))for x in ranges],
                    [-x+tanh(x) for x in ranges],
               [exp(-x**4/4) for x in ranges],
                    [-x**3 for x in ranges],
               [(x**2+5)**-3 for x in ranges],
                    [(-6*x)/(x**2)+5 for x in ranges]].T

    for i in range(0,outputs.shape[0],2):
        ax = subplot(outputs.shape[0], 1, i + 1)
        plot(outputs[i])
        plot(outputs[i+1])
        ax.set_xticks([])
        ax.set_yticks([])
        
    show()

def test_whitening():
    data = np.random.randn(3, 1000000)*1000
    white_data = whiten(data)
    white_covariance = np.cov(white_data)
    #white_covariance = np.diag(np.diag(white_covariance)) this will always end up as a diagonal matrix, thus a bad test
    ax = imshow(white_covariance, cmap='gray', interpolation='nearest')
    show()

def test_power():
    m = np.array([[1,2], [3,4]])
    print m**-0.5

if __name__ == '__main__':
    test_whitening()
    #test_power()
    #plot_functions()

