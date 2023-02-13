from keras.datasets import fashion_mnist

def load_data(set):
    # load the Fashion MNIST dataset
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
    if set == 'train':
        # summarize loaded train set
        # print("X_train=%s, y_train=%s" % (X_train.shape, y_train.shape))
        return X_train, y_train
    else: # set == 'test'
        # summarize loaded test set
        # print("X_test=%s, y_test=%s" % (X_test.shape, y_test.shape))
        return X_test, y_test

def preprocess_data(X):
    # normalize the pixel values
    X = X / 255.0
    # flatten each of the images into 1D arrays since we are dealing with only Dense layers and no CNN layers
    X = X.reshape(-1, X.shape[1] * X.shape[2])
    # transpose the data so that the number of rows in the data matrix is equal to the number of columns in the weight matrices,
    # and the matrix multiplication between them is successful.
    # after transpose each column of the data matrix will signify the pixel values in each image,
    # and we will get to know about the weight matrix while defining the neural network architecture.
    X = X.T
    return X