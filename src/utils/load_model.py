import pickle

# load the trained model parameters: weights and biases
def load_model_params(model_path):
    params_file = open(model_path,"rb")
    params = pickle.load(params_file)
    W1, b1, W2, b2, W3, b3 = params
    params_file.close()
    return W1, b1, W2, b2, W3, b3