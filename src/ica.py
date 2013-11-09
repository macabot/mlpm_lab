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
    """ 
    Generating data test function
    """

    # create signals
    signal_length = 500
    t = np.linspace(0, 1, signal_length)
    S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25), np.random.randn(t.size), np.random.rand(t.size)].T

    # plots signals
    plot_signals(S)
    plot_histograms(S)

    show()
    return S

def whiten(data):
    """
    whitening
    """

    # get cov of zero mean data
    mean = np.mean(data)
    data -= mean
    covariance = np.cov(data)

    # The columns of phi are the eigenvectors of the covariance matrix.
    phi = np.linalg.eig(covariance)[1]

    # create lamda
    diag_lambda = np.diag(np.dot(np.dot(phi.T, covariance), phi))
    return np.dot(np.dot(np.diag(diag_lambda**-0.5), phi.T), data)
    
def plot_functions():
    """
    Plots the four activation functions calculated in assignments
    """

    figure()

    # create functions
    ranges = range(0,100)
    ranges = np.linspace(-1,1,100)
    outputs = np.c_[[1/cosh(x) for x in ranges],
                    [-tanh(x) for x in ranges],
               [exp(((-x**2)/2) + np.log(cosh(x)))for x in ranges],
                    [-x+tanh(x) for x in ranges],
               [exp(-x**4/4) for x in ranges],
                    [-x**3 for x in ranges],
               [(x**2+5)**-3 for x in ranges],
                    [(-6*x)/(x**2)+5 for x in ranges]].T

    # plot functions
    for i in range(0,outputs.shape[0],2):
        ax = subplot(outputs.shape[0]*2, 2, i + 1)
        plot(outputs[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax = subplot(outputs.shape[0]*2, 2, i + 2)
        plot(outputs[i+1])
        ax.set_xticks([])
        ax.set_yticks([])
        
    show()

def test_whitening():
    """
    Tests the whitening function
    """

    data = np.random.randn(3, 1000000)*1000
    white_data = whiten(data)
    white_covariance = np.cov(white_data)
    white_covariance = np.diag(np.diag(white_covariance)) 
    ax = imshow(white_covariance, cmap='gray', interpolation='nearest')
    show()
    
def ICA_mod(data, activation_function, learning_rate):
    """
    Independent Component Analysis
    TODO fix
    """
    
    # holds our best guess of the correct weights for demixing
    demixer = _sym_decorrelation(random_nonsingular_matrix(len(data)))

    # holds the difference between the new weights and the old
    difference = float('inf')

    # defines the limit of difference between old and new weights
    max_diff = 1e-10

    # whiten the data
    data , _= whiten2(data)
    n, p = data.shape
    it = 0
    while difference > max_diff and it <  15000:
        # put data through a linear mapping
        linmap_data = np.dot(demixer, data)
        # put it through a nonlinear map
        nonlinmap_data = activation_function(linmap_data)
        # put it back through W
        data_prime = np.dot(demixer.T, linmap_data)
        # adjust the weights
        demixer1 = np.dot(nonlinmap_data, data.T)/float(p) - learning_rate * np.dot(np.diag(data_prime.mean(axis=1)), demixer)
        demixer1 = _sym_decorrelation(demixer1)
        difference = max(abs(abs(np.diag(np.dot(demixer1,demixer.T)))-1))
        demixer = demixer1
        it +=1 
        #print(demixer)
        
    return demixer
    
def whiten2(X):
    # Centering the columns (ie the variables)
    X = X - X.mean(axis=-1)[:, np.newaxis]

    # Whitening and preprocessing by PCA
    u, d, _ = linalg.svd(X, full_matrices=False)
    del _
    components = min(X.shape)
    K = (u/d).T[:components] # see (6.33) p.140
    del u, d
    X1 = np.dot(K, X) # see (13.6) p.267 Here X1 is white and data
    # in X has been projected onto a subspace by PCA
    return X1, K
    
def _sym_decorrelation(W):
    """ Symmetric decorrelation """
    K = np.dot(W, W.T)
    s, u = linalg.eigh(K)
    # u (resp. s) contains the eigenvectors (resp. square roots of
    # the eigenvalues) of W * W.T
    u, W = [np.asmatrix(e) for e in (u, W)]
    W = (u * np.diag(1.0/np.sqrt(s)) * u.T) * W # W = (W * W.T) ^{-1/2} * W
    return np.asarray(W)

def ICA(data, activation_function, learning_rate):
    """
    Independent Component Analysis
    TODO fix
    """
    
    # holds our best guess of the correct weights for demixing
    demixer = random_nonsingular_matrix(len(data))

    # holds the difference between the new weights and the old
    difference = float('inf')

    # defines the limit of difference between old and new weights
    max_diff = 0.01

    # whiten the data
    data = whiten(data)

    while difference > max_diff:
        # put data through a linear mapping
        linmap_data = np.dot(demixer, data)
        # put it through a nonlinear map
        nonlinmap_data = activation_function(linmap_data)
        # put it back through W
        data_prime = np.dot(demixer.T, linmap_data)
        # adjust the weights
        demixer_diff = learning_rate * \
                       (np.dot(nonlinmap_data, data_prime.T))
        difference = np.sum(np.absolute(demixer_diff))
        demixer += demixer_diff
        print(demixer)
        
    return demixer
    
def test_ICA():
    """
    Test the ICA function
    """

    # the activation function used in ica and its learning rate
    activation_function = (lambda a: -tanh(a))
    learning_rate = 0.00001

    # create data
    data = generate_data()  
    
    # remove random stuff to make stuff easier (also to check..)
    data = np.delete(data, -1, 0)
    data = np.delete(data, -1, 0)

    # mix data
    mixer = np.random.randn(data.shape[0], data.shape[0])
    mixed_data = make_mixtures(data, np.random.randn(data.shape[0], data.shape[0]))

    # perform ICA
    demixer = ICA(mixed_data, activation_function, learning_rate)

    # compare data
    _, K = whiten2(mixed_data)
    #demixed_data = np.dot(demixer,  mixed_data)
    demixed_data = np.dot(np.dot(demixer, K), mixed_data)
    plot_signals(whiten(data))
    plot_signals(demixed_data)

    show()

def test_power():
    """
    Tests to the power behaviour op numpy
    """

    m = np.array([[1,2], [3,4]])
    print m**-0.5

if __name__ == '__main__':
    test_ICA()
    #test_whitening()
    #test_power()
    #plot_functions()

