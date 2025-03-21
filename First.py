import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# Încărcăm datele 
data = pd.read_csv('1.csv')
data_Set = pd.DataFrame(data)
data_1 = pd.read_csv('ml_insurance_challenge.csv')
data_Set2 = pd.DataFrame(data_1)

# Afișăm datele pentru verificare
print("Primele rânduri din primul set de date:")
print(data.head())
print("\nPrimele rânduri din al doilea set de date:")
print(data_1.head())




