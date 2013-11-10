from numpy import *
from matplotlib.pyplot import *
import scipy.io.wavfile

def save_wav(data, out_file, rate):
    scaled = np.int16(data / np.max(np.abs(data)) * 32767)
    scipy.io.wavfile.write(out_file, rate, scaled)          

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
    S = np.c_[sawtooth(t), sine_wave(t, 0.3), square_wave(t, 0.4), triangle_wave(t, 0.25)].T

    # plots signals
    # plot_signals(S)
    # plot_histograms(S)

    # show()
    return S

def whiten(data):
    """
    whitening
    """

    # get cov of zero mean data
    mean = np.mean(data, axis=1)[np.newaxis,:].T
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

def ICA(data, activation_function, learning_rate):
    """
    Independent Component Analysis
    """

    # holds the difference between the new weights and the old
    difference = float('inf')
    # defines the limit of difference between old and new weights
    max_diff = 1e-3
    # holds our best guess of the correct weights for demixing
    demixer = random_nonsingular_matrix(data.shape[0])
    # holds the difference in weights
    diff = float('inf')

    # whiten the data
    data = whiten(data)

    # contribution of each data point
    N = 1./data.shape[1]
    it = 0 # current iteration

    print("Running activation function: " + str(activation_function))
    #while difference > max_diff and it < 5000: 
    while it < 15000 and diff > max_diff:
        # put data through a linear mapping
        linmap_data = np.dot(demixer, data)

        # put it through a nonlinear map
        nonlinmap_data = activation_function(linmap_data)

        # put it back through W
        data_prime = np.dot(demixer.T, linmap_data)

        # adjust the weights
        demixer_diff = demixer + N * \
                       (np.dot(nonlinmap_data, data_prime.T))
        diff = np.sum(np.absolute(demixer_diff))
        demixer += learning_rate * demixer_diff

        it += 1

        if it % 30 == 0:
            print("iteration: " + str(it) + ", diff: " + str(diff))

    return np.dot(demixer, data)
     
def test_ICA():
    """ Tests the ICA method """
    # define parameters
    learning_rate = 0.1
    activation_function = (lambda a: -a + tanh(a))

    # create data
    data = generate_data()
    mixed_data = np.dot(random_nonsingular_matrix(data.shape[0]), data)

    # perform ica
    sources = ICA(mixed_data, activation_function, learning_rate)

    # plot results
    plot_signals(sources)
    show()

def test_power():
    """
    Tests to the power behaviour op numpy
    """

    m = np.array([[1,2], [3,4]])
    print m**-0.5

def test_activations():
    """
    An exercise of the notebook, check the performance of each activation function
    """

    # parameter constants
    learning_rate = 0.1

    # the activation functions to test
    act_funcs = [(lambda a: -tanh(a)), (lambda a: -a + tanh(a)), \
                 (lambda a: -a**3),(lambda a: - ( (6*a)/(a**2+5) ))]
   
    # generate data (is the same for each test)
    data = generate_data()
    mixed_data = np.dot(random_nonsingular_matrix(data.shape[0]), data)
   
    # perform ica with act func
    for act_func in act_funcs:
        source = ICA(mixed_data, act_func, learning_rate)

        plot_signals(source)

    show() 

def demix_audio():
    """
    The audio demixing assignment in notebook
    todo: something goes wrong with reading in files, as whitening produces shit
    """

    # learning rate used by ICA
    learning_rate = 0.1
    # the activation functions to test
    act_funcs = [(lambda a: -tanh(a)), (lambda a: -a + tanh(a)), \
                (lambda a: -a**3),(lambda a: - ( (6*a)/(a**2+5) ))]
    # holds the eventual demixed audio files
    demixed = []

    # Load audio sources
    source_files = ['X0.wav', 'X1.wav', 'X2.wav', 'X3.wav', 'X4.wav']
    wav_data = []
    sample_rate = None
    for f in source_files:
        sr, data = scipy.io.wavfile.read('../../' + f)
        if sample_rate is None:
            sample_rate = sr
        else:
            assert(sample_rate == sr)
        wav_data.append(data[:190000]) 

    # Create source and measurement data
    S = np.c_[wav_data]

    # perform ica with act func on mixed audio
    for act_func in act_funcs:
        demixed.append(ICA(S, act_func, learning_rate))
 
    # save files away
    for i in range(demixed.shape[0]):
        for j in range(demixed.shape[1]):
            save_wav('../../' + demixed[i, :], 'demixed' + str(i) + str(j) + '.wav', sample_rate)
                
          

if __name__ == '__main__':
    #test_whitening()
    #test_power()
    #plot_functions()
    #test_ICA()
    #test_activations()
    demix_audio()

