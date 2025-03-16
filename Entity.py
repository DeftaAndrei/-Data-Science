
import pandas as pd
import numpy as np
import pyarrow
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from fuzzywuzzy import fuzz
from tensorflow.python.framework import ops




filename ="C:/Users/Andrei-RobertDEFTA/Desktop/Veridion/Entity/veridion_entity_resolution_challenge.snappy.parquet"


df = pd.read_parquet(filename)

# Preprocesarea datelor pentru identificarea companiilor unice
def preprocess_company_data(df):
    # Curățarea și standardizarea numelor de companii
    df['company_name_clean'] = df['company_name'].str.lower().str.strip()
    
    # Crearea unui identificator unic bazat pe nume și alte atribute
    df['company_id'] = df.apply(
        lambda x: f"{x['company_name_clean']}_{x['address']}_{x['website']}", 
        axis=1
    )
    
    return df

# Identificarea și gruparea companiilor duplicate
def identify_unique_companies(df):
    # Preprocesarea datelor
    df_processed = preprocess_company_data(df)
    
    # Gruparea după identificatorul unic
    grouped_companies = df_processed.groupby('company_id').agg({
        'company_name': 'first',
        'address': 'first',
        'website': 'first',
        'id': lambda x: list(x)  # Păstrăm toate ID-urile originale
    }).reset_index()
    
    # Adăugăm numărul de duplicate pentru fiecare companie
    grouped_companies['duplicate_count'] = grouped_companies['id'].apply(len)
    
    print(f"Număr inițial de înregistrări: {len(df)}")
    print(f"Număr de companii unice identificate: {len(grouped_companies)}")
    
    return grouped_companies

# Procesarea și afișarea rezultatelor
unique_companies = identify_unique_companies(df)

# Afișăm primele câteva companii unice cu numărul lor de duplicate
print("\nPrimele 5 companii unice cu numărul lor de duplicate:")
print(unique_companies[['company_name', 'duplicate_count']].head())

# Salvăm rezultatele într-un nou fișier
output_file = "unique_companies.csv"
unique_companies.to_csv(output_file, index=False)
print(f"\nRezultatele au fost salvate în {output_file}")

# Date de intrare pentru poarta XOR
x = np.array(([0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]), dtype=float)
# Ieșirile așteptate pentru poarta XOR
y = np.array(([0],[1],[1],[0],[1],[0],[0],[1]), dtype=float)

# Definim variabila pentru salvarea valorilor de pierdere
lossFile = open("SumSquaredLossList.txt", "w")
# Date pentru predicție
xPredicted = np.array(([0,0,1]), dtype=float)

class Neural_Network(object):
    def __init__(self):
        self.inputLayerSize = 3
        self.outputLayerSize = 1
        self.hiddenLayerSize = 4  # mărimea stratului ascuns 
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize)
        self.lossList = []

    def feedForward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.activationSigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        output = self.activationSigmoid(self.z3)
        return output

    def backwardPropagate(self, X, y, output):
        self.o_error = y - output 
        # Aplicăm derivata funcției de activare pentru a calcula delta
        self.o_delta = self.o_error * self.activationSigmoidPrime(output)
        self.z2_error = self.o_delta.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.activationSigmoidPrime(self.z2)
        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def trainNetwork(self, X, y):
        output = self.feedForward(X)
        self.backwardPropagate(X, y, output)
        loss = np.mean(np.square(y - output))
        self.lossList.append(loss)
        return loss

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

# Inițializăm rețeaua neuronală  desi ca sa fie cu adevarat  buna banuiesc ca trebuie antrenata cu valorii mult mai mari
myNeuralNetwork = Neural_Network()
trainingEpochs = 1000

# Antrenăm rețeaua neuronală
for i in range(trainingEpochs):
    loss = myNeuralNetwork.trainNetwork(x, y)
    if i % 100 == 0:
        print(f"Epoca {i}, Pierdere: {loss}")
        myNeuralNetwork.saveSumSquaredLossList(i, loss)

# Salvăm ponderile finale
myNeuralNetwork.saveWeights()

# Facem predicții
myNeuralNetwork.predictOutput()

# Convertim datele procesate în tensori pentru utilizare în TensorFlow
# Extragem doar primele 3 coloane numerice pentru simplificare
numeric_data = unique_companies.select_dtypes(include=[np.number]).iloc[:, :3]
if len(numeric_data.columns) < 3:
    print("Nu sunt suficiente coloane numerice, adăugăm coloane suplimentare")
    # Adăugăm coloane suplimentare dacă nu avem suficiente
    for i in range(3 - len(numeric_data.columns)):
        numeric_data[f'extra_col_{i}'] = 0

tensor_data = tf.convert_to_tensor(numeric_data.values, dtype=tf.float32)

print("Forma tensorului de date procesate:", tensor_data.shape)

# Creăm un model de rețea neuronală cu TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(3,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compilăm modelul
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Creăm etichete artificiale pentru demonstrație
tf_y = np.random.randint(0, 2, size=(len(tensor_data), 1))

# Antrenăm modelul
model.fit(tensor_data, tf_y, epochs=10)  # Reducem numărul de epoci pentru demonstrație

# Facem predicții
predictions = model.predict(tensor_data)

# Afișăm primele 5 predicții   
print("\nPrimele 5 predicții:")
print(predictions[:5])
#Sistemul nu merge foarte bine dar este un inceput
