from utils.dataset import *
from utils.save import *
from utils.fit import *

# load and preprocess the dataset
X_train, y_train, X_test, y_test = load_data()
X_train, X_test = preprocess_data(X_train, y_train)

# fit the defined neural network model with the training data and get the final weights and biases
W1, b1, W2, b2, W3, b3 = gradient_descent(X_train, y_train, 0.10, 2000)

# save the trained model parameters: weights and biases
save_params(W1, b1, W2, b2, W3, b3)