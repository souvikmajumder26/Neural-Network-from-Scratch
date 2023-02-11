import numpy as np

# the neural network architecture: one input layer, two hidden layers, and one output layer

# initialize the weights and biases with random values
def init_params(X, units_layer_h1, units_layer_h2, units_layer_output):

    # n = number of pixels in each image
    n, _ = X.shape

    # initialization of weights can be done either by rand or randn:
    # > rand gives values from a uniform distribution in range [0, 1) with mean 0.5,
    # we subtract the values by 0.5 to make the mean 0 to maintain symmetry in the network.
    # this helps in preventing saturation in the activation function, which can happen when weights are initialized to large values.
    # additionally, a symmetrical distribution ensures that the activations at the output layer will be roughly balanced,
    # avoiding early convergence to a sub-optimal solution.
    # > randn gives values from a standard normal distribution with mean 0 and standard deviation 1,
    # but it is not bounded due to which convergence or improvement in performance takes longer.
    # > initializing the weights to zero will lead all the neurons to learn the same features, which is not expected.
    
    # the shape of weight matrix between the input layer and the first hidden layer:
    # (number of units/neurons in the layer, number of pixels in each image, i.e., number of rows in the input data matrix).
    # for the fashion-mnist dataset, each image is 28*28, therefore the flattened image size is 768 which was assigned to the variable 'n'
    W1 = np.random.rand(units_layer_h1, n) - 0.5
    # initialization of biases can be done either by rand or zeros:
    # > any of the options is feasible since the non-zero weights will not let the gradients vanish

    # the shape of bias matrix for the first hidden layer:
    # (number of units/neurons in the layer, 1).
    b1 = np.random.rand(units_layer_h1, 1)
    # weight matrix between the first and the second hidden layer
    W2 = np.random.rand(units_layer_h2, units_layer_h1) - 0.5
    # bias matrix for the neurons in the second hidden layer
    b2 = np.random.rand(units_layer_h2, 1)
    # weight matrix between the second and the third hidden layer
    W3 = np.random.rand(units_layer_output, units_layer_h2) - 0.5
    # bias matrix for the nerons in the third hidden layer
    b3 = np.random.rand(units_layer_output, 1)
    return W1, b1, W2, b2, W3, b3

# define the activation function for the hidden layers
def relu(Z):
    return np.maximum(0, Z)

# define the derivative of the activation function for the hidden layers
def relu_deriv(Z):
    return Z > 0

# define the activation function for the output layer
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# define the forward propagation
def forward(W1, b1, W2, b2, W3, b3, X):
    # calculate the activations for the first hidden layer
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    # calculate the activations for the second hidden layer
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    # calculate the activations for the output layer
    Z3 = np.dot(W3, A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3

# one hot encoding
def one_hot(y):
    one_hot_y = np.zeros((y.size, y.max() + 1))
    one_hot_y[np.arange(y.size), y] = 1
    # the above line is similar to:
    # for i, j in enumerate(y):
    #   one_hot_y[i][j] = 1
    one_hot_y = one_hot_y.T
    return one_hot_y

# define the backward propagation
def backward(W1, Z1, A1, W2, Z2, A2, W3, Z3, A3, X, y):
    # m = number of images in the train set
    _, m = X.shape
    # get one hot encoded version of the labels
    one_hot_y = one_hot(y)
    # calculate the change in weights and biases for the output layer
    dZ3 = A3 - one_hot_y
    dW3 = 1 / m * np.dot(dZ3, A2.T)
    db3 = 1 / m * np.sum(dZ3)
    # calculate the change in weights and biases for the second hidden layer
    dZ2 = np.dot(W3.T, dZ3) * relu_deriv(Z2)
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # calculate the change in weights and biases for the first hidden layer
    dZ1 = np.dot(W2.T, dZ2) * relu_deriv(Z1)
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2, dW3, db3

# update the parameters
def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3