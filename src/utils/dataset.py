from keras.datasets import fashion_mnist

def load_data():
    # load the Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    # summarize loaded dataset
    print('Train: X=%s, y=%s' % (X_train.shape, y_train.shape))
    print('Test: X=%s, y=%s' % (X_test.shape, y_test.shape))
    return X_train, y_train, X_test, y_test

def preprocess_data(X_train, X_test):
    # normalize the pixel values
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    # flatten each of the images into 1D arrays since we are dealing with only Dense layers and no CNN layers
    X_train = X_train.reshape(-1, X_train.shape[1] * X_train.shape[2])
    X_test = X_test.reshape(-1, X_test.shape[1] * X_test.shape[2])
    # transpose the data so that the number of rows in the data matrix is equal to the number of columns in the weight matrices,
    # and the matrix multiplication between them is successful.
    # after transpose each column of the data matrix will signify the pixel values in each image,
    # and we will get to know about the weight matrix while defining the neural network architecture.
    X_train = X_train.T
    X_test = X_test.T
    return X_train, X_test