import csv 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Setăm stilul pentru grafice
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Încărcăm datele
print("Încărcarea datelor...")
taxonomy_df = pd.read_csv('1.csv')
companies_df = pd.read_csv('ml_insurance_challenge.csv')

# Afișăm datele pentru verificare
print("Primele rânduri din primul set de date:")
print(taxonomy_df.head())
print("\nPrimele rânduri din al doilea set de date:")
print(companies_df.head())

# 1. Analiza taxonomiei
print("\nAnalizarea taxonomiei...")
total_labels = len(taxonomy_df)

# Creăm primul grafic - Distribuția etichetelor din taxonomie
plt.figure(figsize=(15, 6))
plt.title('Distribuția Etichetelor din Taxonomie')
taxonomy_words = ' '.join(taxonomy_df['label']).lower().split()
word_freq = Counter(taxonomy_words).most_common(20)
words, freqs = zip(*word_freq)
plt.bar(words, freqs)
plt.xticks(rotation=45, ha='right')
plt.ylabel('Frecvența')
plt.tight_layout()
plt.savefig('taxonomy_distribution.png')
plt.close()

# 2. Analiza companiilor
print("\nAnalizarea companiilor...")

# Analiza lungimii descrierilor
description_lengths = companies_df['Company Description'].str.len()
plt.figure(figsize=(10, 6))
plt.title('Distribuția Lungimii Descrierilor Companiilor')
plt.hist(description_lengths, bins=50, edgecolor='black')
plt.xlabel('Lungimea descrierii (caractere)')
plt.ylabel('Număr de companii')
plt.tight_layout()
plt.savefig('description_lengths.png')
plt.close()

# 3. Analiza tag-urilor
if 'Business Tags' in companies_df.columns:
    print("\nAnalizarea tag-urilor...")
    # Extragem și numărăm tag-urile
    all_tags = []
    for tags in companies_df['Business Tags'].dropna():
        all_tags.extend([tag.strip() for tag in str(tags).split(',')])
    
    tag_counts = Counter(all_tags).most_common(15)
    tags, counts = zip(*tag_counts)
    
    plt.figure(figsize=(12, 6))
    plt.title('Top 15 Cele Mai Comune Tag-uri de Business')
    plt.bar(tags, counts)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Frecvența')
    plt.tight_layout()
    plt.savefig('business_tags.png')
    plt.close()

# 4. Analiza sectoarelor și categoriilor
if 'Sector' in companies_df.columns:
    print("\nAnalizarea sectoarelor...")
    sector_counts = companies_df['Sector'].value_counts().head(10)
    
    plt.figure(figsize=(12, 6))
    plt.title('Top 10 Sectoare')
    sector_counts.plot(kind='bar')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Număr de companii')
    plt.tight_layout()
    plt.savefig('sector_distribution.png')
    plt.close()

# 5. Analiza rezultatelor clasificării
print("\nAnalizarea rezultatelor clasificării...")
# Rulăm clasificatorul
from Read import CompanyClassifier

classifier = CompanyClassifier()
classifier.fit(companies_df)
predictions = classifier.predict(companies_df)
companies_df['predicted_labels'] = predictions

# Calculăm statistici despre predicții
prediction_counts = pd.Series([label.split(', ')[0] for label in predictions if label != 'Unclassified'])
top_predictions = prediction_counts.value_counts().head(15)

plt.figure(figsize=(15, 6))
plt.title('Top 15 Cele Mai Frecvente Etichete Prezise')
plt.bar(range(len(top_predictions)), top_predictions.values)
plt.xticks(range(len(top_predictions)), top_predictions.index, rotation=45, ha='right')
plt.ylabel('Număr de companii')
plt.tight_layout()
plt.savefig('prediction_distribution.png')
plt.close()

# 6. Analiza scorurilor de similaritate
print("\nAnalizarea scorurilor de similaritate...")
# Calculăm scorurile de similaritate pentru primele 1000 de companii
sample_size = min(1000, len(companies_df))
similarity_scores = []

for idx, row in companies_df.head(sample_size).iterrows():
    features = classifier.extract_features(row)
    company_vector = classifier.create_weighted_vector(features)
    similarities = [classifier.calculate_similarity(company_vector, tax_vec) 
                   for tax_vec in classifier.taxonomy_vectors]
    similarity_scores.extend(similarities)

plt.figure(figsize=(10, 6))
plt.title('Distribuția Scorurilor de Similaritate')
plt.hist(similarity_scores, bins=50, edgecolor='black')
plt.xlabel('Scor de similaritate')
plt.ylabel('Frecvența')
plt.tight_layout()
plt.savefig('similarity_scores.png')
plt.close()

# 7. Matrice de confuzie pentru categorii principale
if 'Category' in companies_df.columns:
    print("\nCrearea matricei de confuzie pentru categorii...")
    predicted_categories = [label.split(', ')[0] if label != 'Unclassified' else 'Unknown' 
                          for label in predictions]
    actual_categories = companies_df['Category'].fillna('Unknown')
    
    # Luăm top 10 categorii pentru vizualizare
    top_categories = pd.Series(actual_categories).value_counts().head(10).index
    
    confusion_data = pd.crosstab(
        pd.Series(actual_categories).map(lambda x: x if x in top_categories else 'Other'),
        pd.Series(predicted_categories).map(lambda x: x if x in top_categories else 'Other')
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(confusion_data, annot=True, fmt='d', cmap='YlOrRd')
    plt.title('Matrice de Confuzie pentru Categoriile Principale')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

