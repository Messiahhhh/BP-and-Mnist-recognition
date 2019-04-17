import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def relu(x):
    # print(x)
    M=[]
    for i in x:
        if(i[0]<0):
            M.append([0.0])
        else:
            M.append(i)
    M=np.array(M)
    # print(M)
    return M
def derivatesig(x):
    return x*(1-x)
def derivatetanh(x):
    return 1-x*x
def derivaterelu(x):
    M=[]
    # print(x)
    for i in x:
        if(i[0]<0):
            M.append([0])
        else:
            M.append([1])
    M=np.array(M)
    # print(M)
    return M