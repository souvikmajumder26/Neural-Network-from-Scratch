import pickle

# save the trained model parameters: weights and biases
def save_model_params(W1, b1, W2, b2, W3, b3):
    params = (W1, b1, W2, b2, W3, b3)
    params_file = open(r"C:\Users\smsou\OneDrive\MY-DEVICE\Sync\Documents\GitHub\Neural-Network-from-Scratch\model\model.pkl","wb")
    pickle.dump(params, params_file)
    params_file.close()
    print("Model parameters saved!")