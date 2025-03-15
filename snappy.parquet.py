from fastparquet import ParquetFile
import numpy as np
import pyarrow 
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python.framework import ops

filename = "C:\Users\Andrei-RobertDEFTA\Desktop\Veridion\Ex1\dt.parquet"
df = pyarrow.parquet.read_table(filename).to_pandas()

x = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
# x fiind poarte de intrare a datelor in retea

y = np.array(([0],[1],[1],[0],[1],[0],[0],[1]), dtype=float)
# y fiind iesirea retelei


xPredicted = np.array(([0,0,1]), dtype=float)

lossFille =open("SumSquaredLossList.parquet" , "w")


class Neural_Network (object):
     def  __init__(self):
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4 # marimea leyerului ascuns 
        self.W1 = np.random.randn(self.inputLayersSize , self.hiddenLayerSize)
        self.W2 = np.random(self.hiddenLayerSize , self.outputLayerSize) 

def feedForward(self, x):
    self.z = np.dot(x, self .W1)
    self.z2_delta = self.activationSigmoind(self.z)
    self.z3 = np.dot(self.o_delta.dot(self.o_delta))
    o = self.o_delta.dot(self.W2.T)
    return o

def backwardPropagete(self, x, y, o):
    self.o_error = y - 0 
    # aplicand astfel incat sa vedem daca la derivare este o eroare cand vine vorba de activare a sistemului 
    self.o_delta = self.o_errore *self.activationSigmoidPrime(o)
    self.z2_delta = self.z2_error*self.activationSigmoidPrime(self.z2)
    self.z2_error = self.o_delta.dot(self. W2.T)
    self.W1 += X.T.dot(self.z2_delta)
    self.W2+= self.z2.T.dot(self.o_delta)


def trainNetwork(self, x, y , o):
    self.yHat = self.forward(x)
    self.loss = self.lossFunction(y, self.yHat)
    self.lossList.append(self.loss)
    self.backward(x, y, o)


def activationSigmoid(self ,s):
    return 1 /(1 + np.exp(-s)) 

def activationSigmoidPrime(self , s):
    return s * (1 - s)


def saveSumSquaredLossList(self,i,error):
    lossFile.write(str(i)+","+str(error)+'\n')

def saveWeights(self):
    np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
    np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")

def predictOutput(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredicted))
    print("Output: \n" + str(self.forward(xPredicted)))

def predictOutput(self):
    print("Predicted data based on trained weights: ")
    print("Input (scaled): \n" + str(xPredicted))
    print("Output: \n" + str(self.forward(xPredicted)))

myNeralNetwork =Neural_Network()
trainingEpochs = 1000

 #Acesta facanduse de minim  100000 de ori pentru a se antrena reteaua corespunzator dar eu personal inca nu am echipament dedicat pentru a face acest lucru
 # si se incearca a optine o eroarecat mai mica dar modelul trebuie antrenat cu date de antrenare cat mai precise si multe dar fix pe ceea ce trebuie sa fie bazate in asa fel sa nu se faca overfitting
 # un sistem complet antrenat va avea o eroare cat mai mica si va fi capabil sa faca predictii cat mai precise
    
for i in range(trainingEpochs):
    print("Epoch:" + str(i) + "\n")
    print("Input: \n" + str(x))
    print ("Expected Output of XOR Gate Neural Network: \n" + str(y)) 
    print ("Actual Output from XOR Gate Neural Network: \n" + str(myNeralNetwork.feedForward(x)))
    # inseamna ca reteaua a fost antrenata si a fost capabila sa faca predictii
    

    Loss =np.mean(np.sqare(y - myNeralNetwork.fedForward(X)))
    myNeralNetwork.SaveSumSquaredLossList(i,Loss)
    print("Sum Squared Loss: \n" + str(Loss))
    print("\n")
    myNeralNetwork.trainNetwork(x, y)
       

myNeralNetwork.saveWeights()
myNeralNetwork.predictOutput()

plt.plot(myNeralNetwork.lossList)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

