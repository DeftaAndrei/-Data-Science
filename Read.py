# Sa recumosc nu am scris tot din cap ma bazez mult pe niste siteri si articole dupa google Academic 
# si am facut un model de clasificare a companiilor in functie de caracteristicile lor

import pandas as pd
import numpy as np
import re
from collections import Counter

class CompanyClassifier:
    def __init__(self, taxonomy_path='1.csv'):
        """Inițializează clasificatorul cu taxonomia dată"""
        self.taxonomy_df = pd.read_csv(taxonomy_path)
        self.vocab = set()
        self.word_weights = {}
        
    def preprocess_text(self, text):
        """Curăță și normalizează textul"""
        if isinstance(text, str):
            
            text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
           
            return re.sub(r'\s+', ' ', text).strip()
        return ''
    
    def extract_features(self, row):
        """Extrage și combină toate caracteristicile relevante într-un singur text"""
        features = []
        
        # Adăugăm descrierea companiei (pondere mare)
        if 'Company Description' in row:
            features.extend([self.preprocess_text(row['Company Description'])] * 3)
            
        # Adăugăm tag-urile de business (pondere medie)
        if 'Business Tags' in row and isinstance(row['Business Tags'], str):
            features.extend([self.preprocess_text(row['Business Tags'])] * 2)
            
        # Adăugăm sectorul și categoria (pondere mică)
        for field in ['Sector', 'Category', 'Niche']:
            if field in row and isinstance(row[field], str):
                features.append(self.preprocess_text(row[field]))
                
        return ' '.join(features)
    
    def build_vocabulary(self, companies_df):
        """Construiește vocabularul și calculează ponderile cuvintelor"""
        print("Construirea vocabularului...")
        
        # Procesăm toate textele pentru a construi vocabularul
        all_texts = []
        for _, row in companies_df.iterrows():
            text = self.extract_features(row)
            all_texts.append(text)
            self.vocab.update(text.split())
            
        # Calculăm IDF (Inverse Document Frequency)
        doc_freq = Counter()
        for text in all_texts:
            doc_freq.update(set(text.split()))
            
        # Calculăm ponderile cuvintelor folosind IDF
        num_docs = len(all_texts)
        self.word_weights = {word: np.log(num_docs / (freq + 1)) 
                           for word, freq in doc_freq.items()}
    
    def create_weighted_vector(self, text):
        """Creează un vector ponderat pentru text"""
        words = text.split()
        vec = np.zeros(len(self.vocab))
        vocab_list = list(self.vocab)
        
        for word in words:
            if word in self.vocab:
                idx = vocab_list.index(word)
                vec[idx] = self.word_weights.get(word, 1.0)
        return vec
    
    def calculate_similarity(self, vec1, vec2):
        """Calculează similaritatea cosinus între doi vectori"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def fit(self, companies_df):
        """Pregătește clasificatorul cu datele de antrenare"""
        print("Începerea antrenării clasificatorului...")
        
        # Construim vocabularul și calculăm ponderile
        self.build_vocabulary(companies_df)
        
        # Preprocesăm taxonomia
        self.taxonomy_df['clean_label'] = self.taxonomy_df['label'].apply(self.preprocess_text)
        self.taxonomy_vectors = np.array([
            self.create_weighted_vector(label) 
            for label in self.taxonomy_df['clean_label']
        ])
        
        print(f"Vocabular construit cu {len(self.vocab)} cuvinte unice")
        return self
    
    def predict(self, companies_df, top_n=3, threshold=0.1):
        """Prezice etichetele pentru companiile date"""
        print("Începerea predicțiilor...")
        predictions = []
        
        for idx, row in companies_df.iterrows():
            if idx % 100 == 0:
                print(f"Procesăm compania {idx}...")
                
            # Extragem și vectorizăm caracteristicile
            features = self.extract_features(row)
            company_vector = self.create_weighted_vector(features)
            
            # Calculăm similaritățile cu toate etichetele
            similarities = [
                self.calculate_similarity(company_vector, tax_vec)
                for tax_vec in self.taxonomy_vectors
            ]
            
            # Selectăm cele mai relevante etichete
            top_indices = np.argsort(similarities)[-top_n:][::-1]
            top_scores = [similarities[i] for i in top_indices]
            top_labels = [
                self.taxonomy_df['label'].iloc[i]
                for i, score in zip(top_indices, top_scores)
                if score > threshold
            ]
            
            predictions.append(', '.join(top_labels) if top_labels else 'Unclassified')
        
        return predictions

def main():
    print("Încărcarea datelor...")
    companies_df = pd.read_csv('ml_insurance_challenge.csv')
    
    # Inițializăm și antrenăm clasificatorul
    classifier = CompanyClassifier()
    classifier.fit(companies_df)
    
    # Facem predicții
    predictions = classifier.predict(companies_df)
    companies_df['predicted_labels'] = predictions
    
    # Analiza rezultatelor
    print("\n=== Statistici generale ===")
    print(f"Număr total de companii: {len(companies_df)}")
    print(f"Număr total de etichete în taxonomie: {len(classifier.taxonomy_df)}")
    
    # Calculăm și afișăm metricile
    classified_companies = sum(1 for p in predictions if p != 'Unclassified')
    classification_rate = classified_companies / len(companies_df) * 100
    
    print("\n=== Metrici de performanță ===")
    print(f"Companii clasificate: {classified_companies}")
    print(f"Rata de clasificare: {classification_rate:.2f}%")
    
    # Distribuția predicțiilor
    print("\n=== Top 10 cele mai frecvente predicții ===")
    print(pd.Series(predictions).value_counts().head(10))
    
    # Salvăm rezultatele
    print("\nSalvarea rezultatelor...")
    companies_df.to_csv('classified_companies_v2.csv', index=False)
    
    # Salvăm metricile
    metrics_df = pd.DataFrame({
        'Metric': [
            'Total Companies',
            'Total Labels',
            'Vocabulary Size',
            'Classification Rate',
            'Classified Companies'
        ],
        'Value': [
            len(companies_df),
            len(classifier.taxonomy_df),
            len(classifier.vocab),
            f"{classification_rate:.2f}%",
            classified_companies
        ]
    })
    metrics_df.to_csv('classification_metrics_v2.csv', index=False)
    
    print("\nProcesul de clasificare s-a încheiat cu succes!")
    print("Rezultatele au fost salvate în 'classified_companies_v2.csv'")
    print("Metricile au fost salvate în 'classification_metrics_v2.csv'")

if __name__ == "__main__":
    main()


