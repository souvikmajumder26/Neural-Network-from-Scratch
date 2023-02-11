import pickle

# save the trained model parameters: weights and biases
def save_model_params(W1, b1, W2, b2, W3, b3, model_path):
    params_file = open(model_path,"wb")
    params = (W1, b1, W2, b2, W3, b3)
    pickle.dump(params, params_file)
    params_file.close()