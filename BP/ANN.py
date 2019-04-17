import numpy as np
import math
import matplotlib.pyplot as plt
from . import ActivateFuc
from mpl_toolkits.mplot3d import Axes3D
class ANN:
    def __init__(self, layer, sample_in, sample_out, lrates, cycles):
        self.sample_in = np.array(sample_in) 
        self.sample_out = np.array(sample_out)  
        self.in_sample=""
        self.out_sample=""
        # print(np.shape(self.sample_in[0]))
        # print(self.sample_in)
        # print(self.sample_out)
        self.cycles = cycles
        self.layer = layer
        self.lrates = lrates
        self.num_layers = len(layer)
        print(layer)
        print("ss")
        self.biases = [np.random.randn(y, 1) for y in layer[1:]]
        print("ss")
        # print("bisa",self.biases)
        self.neu = []  
        self.weights = [np.random.randn(x, y) for x, y in
                        zip(self.layer[:-1], self.layer[1:])] 
        for i in range(0, len(self.weights)):
            self.weights[i] = self.weights[i].T
        # print("weig", self.weights)
    def forwar(self, index):
        label = 0
        # print(self.weights)
        # print(self.biases)
        Data = ''
        for i in range(0, len(self.layer) - 1):
            if (label == 0):
                Data = np.array(self.sample_in[index])
                #    print(Data)
                # print(self.weights[i])
                # print(self.weights[i].shape)
                # print(Data)
                # print(Data.shape)
                # print(np.dot(self.weights[i],Data))
                Data = np.dot(self.weights[i], Data)
                # print(Data)
                Data = Data + self.biases[i]
                # print(Data)
                Data = ActivateFuc.sigmoid(Data)  # Sigmoid function
                # print(Data)
                self.neu.append(Data)
                label = 1
            else:
                if(i!=(len(self.layer)-2)):
                    Data = np.dot(self.weights[i], Data)
                # print(Data)
                    Data = Data + self.biases[i]
                    Data = ActivateFuc.sigmoid(Data) # Sigmoid function
                else:
                    Data = np.dot(self.weights[i], Data)
                    Data = Data + self.biases[i]
                self.neu.append(Data)
                # print("bitch")
        # print("neu",self.neu)
        # print(self.neu)
        return Data
    def InitTest(self,in_sample):
        self.in_sample=in_sample
    def TestForwar(self, index):
        label = 0
        # print(self.weights)
        # print(self.biases)
        Data = ''
        for i in range(0, len(self.layer) - 1):
            if (label == 0):
                Data = np.array(self.in_sample[index])
                #    print(Data)
                # print(self.weights[i])
                # print(self.weights[i].shape)
                # print(Data)
                # print(Data.shape)
                # print(np.dot(self.weights[i],Data))
                Data = np.dot(self.weights[i], Data)
                # print(Data)
                Data = Data + self.biases[i]
                # print(Data)
                Data = ActivateFuc.sigmoid(Data)  # Sigmoid function
                # print(Data)
                self.neu.append(Data)
                label = 1
            else:
                if(i!=(len(self.layer)-2)):
                    Data = np.dot(self.weights[i], Data)
                # print(Data)
                    Data = Data + self.biases[i]
                    Data = ActivateFuc.sigmoid(Data) # Sigmoid function
                else:
                    Data = np.dot(self.weights[i], Data)
                    Data = Data + self.biases[i]
                self.neu.append(Data)
        return Data
    def backwar(self, index):
        deltaw = [] 
        d = []  
        label = 0
        # print(self.weights)
        '''for i in range(len(self.weights)-1,-1,-1):
            # print(i)
            if(label==0):
                # print("1",self.neu[i])
                # print("2",1-self.sample_out[index])
                # print("3",self.neu[i]-self.sample_out[index])
                d.append(self.neu[i]*(1-self.neu[i])*(self.neu[i]-self.sample_out[index]))#是1-self.neu[i]
                # print("d[-1]",d[-1])
                deltaw.append(-self.lrates*np.dot(self.neu[i-1].T,d[-1]))#存疑，neu为i-1,一直减去前一个
                # print("deltaw",deltaw[-1])
                label=1
            else:
                print("1",self.neu[i])
                print("2",1-self.neu[i])
                print(np.dot(d[-1], self.weights[i + 1].T))
                d.append(self.neu[i]*(1-self.neu[i])*np.dot(d[-1],self.weights[i+1].T))
                print(d[-1])
                if(i==0):
                    deltaw.append(-self.lrates*np.dot(self.sample_in[index].T,d[-1]))#每一层的权值修改量，顺序是反的，由输出层到隐层第一层
                    # print(deltaw[-1])
                else:
                    deltaw.append(-self.lrates * np.dot(self.neu[i-1].T,d[-1]))'''
        for i in range(len(self.neu) - 1, -1, -1):
            # print(i)
            if label == 0:
                # d.append(self.neu[i] * (1 - self.neu[i]) * (self.neu[i] - self.sample_out[index]))
                d.append(self.neu[i] - self.sample_out[index])
                # print(d)
                label = 1
            else:
                # print("fuck",self.weights[i+1])
                # print(d[-1])
                d.append(ActivateFuc.derivatesig(self.neu[i])*np.dot(self.weights[i + 1].T, d[-1]))
                # print(d)

        for i in range(0, len(self.weights)):
            if (i == 0):
                # print("w",self.weights)
                # print("b",self.biases)
                # print(d[len(d)-1-i])
                # print("kan",np.dot(d[len(d)-1-i],np.array([self
                #.sample_in[index]]).T))
                # print(self.sample_in[index].T.shape)
                # print(np.dot(d[len(d) - 1 - i], np.array(self.sample_in[index]).T))
                self.weights[i] = self.weights[i] - self.lrates * (
                            np.dot(d[len(d) - 1 - i], np.array(self.sample_in[index]).T))
                self.biases[i] = self.biases[i] - self.lrates * d[len(d) - 1 - i]
                # print(np.shape(self.weights))
                # print(np.shape(self.biases))
                # print(self.weights)
                # print(self.biases)
                # print("w2",self.weights)
                # print("b2",self.biases)
            else:
                # print("wk",self.weights)
                # print("bk",self.biases)
                self.weights[i] = self.weights[i] - self.lrates * (
                            np.dot(d[len(d) - 1 - i], self.neu[i - 1].T))
                self.biases[i] = self.biases[i] - self.lrates * d[len(d) - 1 - i]
                # print("wlk2",self.weights)
                # print("blk2",self.biases)

            # print("wei",self.weights[i])
            # print("del",deltaw[len(deltaw)-1-i])
            # self.weights[i]=self.weights[i]+deltaw[len(deltaw)-1-i]
            # self.biases[i]=self.biases[i]-self.lrates*d[len(d)-1-i]
        # print("weigh",self.weights)
        # print("biase",self.biases)

    def run(self):
        X = []
        Y = []
        for i in range(0, self.cycles):
            # print("weight",self.weights)
            # print("bias",self.biases)

            Y.append(self.train(i))
            X.append(i)
        return X, Y

    def PlotDemo1(self):
        X, Y = self.run()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(X, Y)
        plt.show()

    def train(self, cycle):
        AVG_loss = 0.0
        Key = []
        for index in range(0, len(self.sample_in)):
            Key = self.forwar(index)
            AVG_loss += (Key[0][0] - self.sample_out[index]) ** 2 / 2
            self.backwar(index)
            self.neu.clear()
        AVG_loss=AVG_loss/len(self.sample_in)
        # print("In %s cycle the loss is %s"%(cycle,AVG_loss))
        return AVG_loss









































