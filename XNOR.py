import numpy as np
import matplotlib.pyplot as plt

x = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)

y = np.array(([0],[1],[1],[0],[1],[0],[0],[1]), dtype=float)

xPredicted = np.array(([0,0,1]), dtype=float)

# Fișier pentru salvarea valorilor de pierdere
lossFile = open("SumSquaredLossList.txt", "w")

class Neural_Network(object):
    def __init__(self):
        # Definirea arhitecturii rețelei
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4  # mărimea stratului ascuns
        
        # Inițializarea ponderilor (weights) cu valori aleatorii
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        
        # Inițializarea listei pentru valorile de pierdere
        self.lossList = []
    
    def feedForward(self, X):
        """
        Propagarea înainte a datelor prin rețea
        
        Args:
            X: Date de intrare
            
        Returns:
            Ieșirea rețelei
        """
        # Calculul pentru primul strat ascuns
        self.z = np.dot(X, self.W1)
        self.z2 = self.activationSigmoid(self.z)
        
        # Calculul pentru stratul de ieșire
        self.z3 = np.dot(self.z2, self.W2)
        output = self.activationSigmoid(self.z3)
        
        return output
    
    def backwardPropagate(self, X, y, output):
        """
        Propagarea înapoi a erorii pentru actualizarea ponderilor
        
        Args:
            X: Date de intrare
            y: Ieșiri așteptate
            output: Ieșiri actuale
        """
        # Calculul erorii la ieșire
        self.o_error = y - output
        
        # Calculul delta pentru stratul de ieșire
        self.o_delta = self.o_error * self.activationSigmoidPrime(output)
        
        # Propagarea erorii către stratul ascuns
        self.z2_error = self.o_delta.dot(self.W2.T)
        
        # Calculul delta pentru stratul ascuns
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)
        
        # Actualizarea ponderilor
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)
    
    def trainNetwork(self, X, y):
        """
        Antrenarea rețelei neuronale
        
        Args:
            X: Date de intrare
            y: Ieșiri așteptate
        """
        # Propagarea înainte
        output = self.feedForward(X)
        
        # Calculul funcției de pierdere (loss function)
        self.loss = self.lossFunction(y, output)
        self.lossList.append(self.loss)
        
        # Propagarea înapoi
        self.backwardPropagate(X, y, output)
    
    def lossFunction(self, y, output):
        """
        Funcția de pierdere - eroarea medie pătratică
        
        Args:
            y: Ieșiri așteptate
            output: Ieșiri actuale
            
        Returns:
            Valoarea pierderii
        """
        return np.mean(np.square(y - output))
    
    def activationSigmoid(self, s):
    
        return 1 / (1 + np.exp(-s))
    
    def activationSigmoidPrime(self, s):
 
        return s * (1 - s)
    
    def saveSumSquaredLossList(self, i, error):

        lossFile.write(str(i) + "," + str(error) + '\n')
    
    def saveWeights(self):
      
        np.savetxt("weightsLayer1.txt", self.W1, fmt="%s")
        np.savetxt("weightsLayer2.txt", self.W2, fmt="%s")
    
    def predictOutput(self):
    
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.feedForward(xPredicted)))

# Crearea și antrenarea rețelei neuronale
myNeuralNetwork = Neural_Network()
trainingEpochs = 1000

# Bucla de antrenare
for i in range(trainingEpochs):
    if i % 100 == 0:  # Afișăm informații doar la fiecare 100 de epoci pentru a reduce output-ul
        print("Epoch: " + str(i))
        print("Input: \n" + str(x))
        print("Expected Output: \n" + str(y))
        print("Actual Output: \n" + str(myNeuralNetwork.feedForward(x)))
        
        loss = myNeuralNetwork.lossFunction(y, myNeuralNetwork.feedForward(x))
        myNeuralNetwork.saveSumSquaredLossList(i, loss)
        print("Sum Squared Loss: \n" + str(loss))
        print("\n")
    
    # Antrenarea rețelei
    myNeuralNetwork.trainNetwork(x, y)

# Salvarea ponderilor și afișarea predicției
myNeuralNetwork.saveWeights()
myNeuralNetwork.predictOutput()

# Vizualizarea evoluției pierderii
plt.figure(figsize=(10, 6))
plt.plot(myNeuralNetwork.lossList)
plt.title('Evoluția funcției de pierdere')
plt.xlabel('Epocă')
plt.ylabel('Pierdere')
plt.grid(True)
plt.savefig('loss_evolution.png')
plt.show()

# Închiderea fișierului
lossFile.close() 
