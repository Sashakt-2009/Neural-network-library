"""This is a self made Neural Network library made by a High School student. I warn you it is definetly full of bugs and may be..., be completly wrong.
If you find any bug or any thing wrong in the code or its implementation plese let me know, I would appriciate your help in any way. If you find this code intresting or 
useful, in any case feel free to use or tinker with it."""

"""This library was made for easier setup of neural networks. The main goal for this library is not better performance, optimization, or coustomizablity:
it"s only goal is for student to better understand, visualize and make neural networks with out knowing the necessary mind boggeling maths."""


"""      !!!!IMPORTANT!!!!  Can anybody review the intitPrams, ForwardProp and the BackProp functions, I geniuenly feel it has somthing wrong in it.     """


import numpy as np 


def input_vec(input):   
    input = np.array(input).flatten().reshape(1, -1)        # convert input list to an array
    return input 

def output_vec(input):
    input = np.array(input).flatten().reshape(1, -1)     # convert output list to an array
    return input


def intitParams(Layer_no: int, Layer_width: int, output_size: int, input_size: int):        # initiates random values [-0.5, 0.5] for weights and biases
    weights = {}                                                                                # NOTE: Layer_no means no. of hidden layers + output layer
    biases = {}                                                                        
                                                                                        

    #input Layer 
    weights["W1"] = np.random.rand(Layer_width, input_size) - 0.5
    biases["B1"] = np.random.rand(Layer_width, 1) - 0.5

    #hidden Layers
    if Layer_no > 2:
        for Layer in range(2,Layer_no):
            weights[f"W{Layer}"] = np.random.rand(Layer_width, Layer_width) - 0.5
            biases[f"B{Layer}"] = np.random.rand(Layer_width, 1) - 0.5

    #output Layer
    weights[f"WY"] = np.random.rand(output_size, Layer_width) - 0.5
    biases[f"BY"] = np.random.rand(output_size, 1) - 0.5

    return weights, biases

  
def activation(activation_fun: str, X):     # activation function with sigmoid ReLU and softmax functions currently available
                                            # NOTE: it returns a (1,1) zero array for any other functions
    if activation_fun == "sigmoid":
        X = 1/ (1 + np.exp(X*-1))
        return X
    
    if activation_fun == "ReLU":
        return np.maximum(0, X)
    
    if activation_fun == "softmax":
        X = np.exp(X) / sum(np.exp(X))
        return X

    
    else:
        return np.zeros((1,1))


def ForwardProp(input_vec, weights, biases, activation_fun1, activation_fun2, Layer_no):            #forward propogation function 
    
    layers = {}

    # A1 --> Z1
    Z =  weights["W1"].dot(input_vec) + biases["B1"]
    layers["A1"] = activation(activation_fun1, Z)
    
    # Z1 --> Z2 --> ..... --> Z(n)
    if Layer_no > 2:
        for Layer in range(2, Layer_no):
            Z = weights[f"W{Layer}"].dot(layers[f"Z{Layer-1}"] if Layer-1 > 1 else layers["A1"]) + biases[f"B{Layer}"]
            layers[f"Z{Layer}"] = activation(activation_fun1, Z)
    del Z

    # Z(n) --> Y
    Y = weights[f"WY"].dot(layers[f"Z{Layer_no-1}"] if Layer_no-1 > 1 else layers["A1"]) + biases[f"BY"]
    layers["Y"] = activation(activation_fun2, Y)
    del Y

    return layers


def deactivate(activation_funn, X):         # findes derivative of the activaiton function
    if activation_funn == "ReLU":
        return X > 0
    
    else:
        return 1
    

def BackProp(values, Layer_no, layers, weights, input_size):        # Back propogatin function

    err_dict = {}
    d = err_dict

    m = input_size   # number of samples

    # Output layer error
    d["Y"] = layers["Y"] - values
    d["WY"] = (1/m) * d["Y"].dot(layers[f"Z{Layer_no-1}"].T if Layer_no-1 > 1 else layers["A1"].T)
    d["BY"] = (1/m) * np.sum(d["Y"], keepdims=True)

    # Hidden layers and input layer error
    for Layer in range(Layer_no-1, 0, -1):
        if Layer > 1:
            d[f"Z{Layer}"] = weights["WY"].T.dot(d["Y"]) * deactivate("ReLU", layers[f"Z{Layer}"])
            d[f"W{Layer}"] = (1/m) * d[f"Z{Layer}"].dot(layers[f"Z{Layer-1}"].T if Layer_no-1 > 1 else layers["A1"].T)
        else:
            d["A1"] = weights["WY"].T.dot(d["Y"]) * deactivate("ReLU", layers["A1"])
            d["W1"] = (1/m) * d[f"A1"].dot(layers["A1"].T)
        d[f"B{Layer}"] = (1/m) * np.sum(d[f"Z{Layer}"] if Layer > 1 else np.sum(d["A1"]), keepdims=True)

    return d


def UpdateGrad(Layer_no, lr ,weights, biases, err_dict):                # Updadtes the gradient based on the err. found during back propogation

    d = err_dict

    #input Layer 
    weights["W1"] = weights["W1"] - lr * d["W1"]
    biases["B1"] = biases["B1"] - lr * d["B1"]

    #hidden Layers
    for Layer in range(2,Layer_no):
        weights[f"W{Layer}"] = weights[f"W{Layer}"] - lr * d[f"W{Layer}"]
        biases[f"B{Layer}"] = biases[f"B{Layer}"] - lr * d[f"B{Layer}"]
    #output Layer
    weights[f"WY"] = weights[f"WY"] - lr * d[f"WY"]
    biases[f"BY"] = biases[f"BY"] - lr * d[f"BY"]

    return weights,biases


def loss(y_true, y_pred):                   # loss function --> calculates the MSE
    return np.mean((y_true - y_pred) ** 2)






if __name__ == "__main__":                  # the main file construct that allows to check if the err. is due to the lib or due to other reasons 
    import pandas as pd 
    input = pd.read_csv("screentime.csv", header=None).squeeze()        
    output = pd.read_csv("marks.csv", header=None).squeeze()            

    input = input_vec(input)
    output = output_vec(output)
    output = output/100


    weights, biases = intitParams(2,1,1, input.shape[0])
    for i in range(1,1000):
        layer_vec = ForwardProp(input, weights, biases, "ReLU", "sigmoid", 2)               #loop run fo 1000 iterations for a best fit model
        err = BackProp(output, 2, layer_vec, weights, 1)
        weights, biases = UpdateGrad(2, 0.001, weights, biases, err)
        losses = loss(output, layer_vec["Y"])
        # if i % 100:
        #     print(losses)
        #     print((layer_vec["Y"]))
        
    
    
    print(f"tained model of loss {losses}")


    




