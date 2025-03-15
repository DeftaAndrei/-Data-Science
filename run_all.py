import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict

# Importăm clasele din celelalte scripturi
from logo_matcher import LogoMatcher
from deep_logo_matcher import DeepLogoMatcher
from ensemble_logo_matcher import EnsembleLogoMatcher

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_all_approaches(parquet_file, eps=0.3, min_samples=2):
    """
    Rulează toate abordările și compară rezultatele
    
    Args:
        parquet_file (str): Calea către fișierul parquet cu datele despre logourile site-urilor
        eps (float): Parametrul epsilon pentru DBSCAN
        min_samples (int): Numărul minim de eșantioane pentru DBSCAN
    """
    # Creăm directorul pentru rezultate
    os.makedirs('results', exist_ok=True)
    
    # Rulăm abordarea tradițională
    logger.info("=== RULĂM ABORDAREA TRADIȚIONALĂ ===")
    start_time = time.time()
    traditional_matcher = LogoMatcher(parquet_file)
    traditional_matcher.load_data()
    traditional_matcher.preprocess_logos()
    traditional_matcher.cluster_logos(eps=eps, min_samples=min_samples)
    traditional_matcher.print_website_groups()
    traditional_matcher.visualize_clusters(output_file='results/traditional_clusters.png')
    traditional_time = time.time() - start_time
    logger.info(f"Abordarea tradițională a durat {traditional_time:.2f} secunde.")
    
    # Rulăm abordarea de învățare profundă
    logger.info("\n=== RULĂM ABORDAREA DE ÎNVĂȚARE PROFUNDĂ ===")
    start_time = time.time()
    deep_matcher = DeepLogoMatcher(parquet_file)
    deep_matcher.load_data()
    deep_matcher.preprocess_logos()
    deep_matcher.cluster_logos(eps=eps, min_samples=min_samples)
    deep_matcher.print_website_groups()
    deep_matcher.visualize_clusters(output_file='results/deep_clusters.png')
    deep_time = time.time() - start_time
    logger.info(f"Abordarea de învățare profundă a durat {deep_time:.2f} secunde.")
    
    # Rulăm abordarea ensemble
    logger.info("\n=== RULĂM ABORDAREA ENSEMBLE ===")
    start_time = time.time()
    ensemble_matcher = EnsembleLogoMatcher(parquet_file)
    ensemble_matcher.load_data()
    ensemble_matcher.preprocess_logos()
    ensemble_matcher.evaluate_clustering()
    ensemble_matcher.print_website_groups()
    ensemble_matcher.visualize_clusters(output_file='results/ensemble_clusters.png')
    ensemble_time = time.time() - start_time
    logger.info(f"Abordarea ensemble a durat {ensemble_time:.2f} secunde.")
    
    # Comparăm rezultatele
    compare_results(traditional_matcher, deep_matcher, ensemble_matcher)

def compare_results(traditional_matcher, deep_matcher, ensemble_matcher):
    """
    Compară rezultatele obținute cu diferite abordări
    
    Args:
        traditional_matcher (LogoMatcher): Instanța clasei LogoMatcher
        deep_matcher (DeepLogoMatcher): Instanța clasei DeepLogoMatcher
        ensemble_matcher (EnsembleLogoMatcher): Instanța clasei EnsembleLogoMatcher
    """
    # Obținem grupurile de site-uri web
    traditional_groups = traditional_matcher.get_website_groups()
    deep_groups = deep_matcher.get_website_groups()
    ensemble_groups = ensemble_matcher.get_website_groups()
    
    # Verificăm dacă avem grupuri valide
    if traditional_groups is None or deep_groups is None or ensemble_groups is None:
        logger.error("Nu există grupuri valide pentru comparație.")
        return
    
    # Creăm un dicționar care mapează fiecare site web la eticheta grupului său pentru fiecare abordare
    traditional_labels = {}
    for group_id, websites in traditional_groups.items():
        for website in websites:
            traditional_labels[website] = group_id
    
    deep_labels = {}
    for group_id, websites in deep_groups.items():
        for website in websites:
            deep_labels[website] = group_id
    
    ensemble_labels = {}
    for group_id, websites in ensemble_groups.items():
        for website in websites:
            ensemble_labels[website] = group_id
    
    # Obținem lista de site-uri web comune
    common_websites = set(traditional_labels.keys()) & set(deep_labels.keys()) & set(ensemble_labels.keys())
    
    # Verificăm dacă avem site-uri web comune
    if not common_websites:
        logger.error("Nu există site-uri web comune pentru comparație.")
        return
    
    # Creăm listele de etichete pentru site-urile web comune
    traditional_common_labels = [traditional_labels[website] for website in common_websites]
    deep_common_labels = [deep_labels[website] for website in common_websites]
    ensemble_common_labels = [ensemble_labels[website] for website in common_websites]
    
    # Calculăm scorul Rand ajustat între diferitele abordări
    trad_deep_score = adjusted_rand_score(traditional_common_labels, deep_common_labels)
    trad_ensemble_score = adjusted_rand_score(traditional_common_labels, ensemble_common_labels)
    deep_ensemble_score = adjusted_rand_score(deep_common_labels, ensemble_common_labels)
    
    # Afișăm rezultatele
    print("\n=== COMPARAREA REZULTATELOR ===\n")
    print(f"Număr de site-uri web comune: {len(common_websites)}")
    print(f"Scorul Rand ajustat între abordarea tradițională și cea de învățare profundă: {trad_deep_score:.4f}")
    print(f"Scorul Rand ajustat între abordarea tradițională și cea ensemble: {trad_ensemble_score:.4f}")
    print(f"Scorul Rand ajustat între abordarea de învățare profundă și cea ensemble: {deep_ensemble_score:.4f}")
    
    # Creăm un tabel cu statistici pentru fiecare abordare
    print("\nStatistici pentru fiecare abordare:")
    print(f"{'Abordare':<20} {'Număr de grupuri':<20} {'Număr de outlier-i':<20}")
    print("-" * 60)
    
    # Calculăm statisticile pentru abordarea tradițională
    trad_n_groups = len([g for g in traditional_groups.keys() if not str(g).startswith('outlier_')])
    trad_n_outliers = len([g for g in traditional_groups.keys() if str(g).startswith('outlier_')])
    print(f"{'Tradițională':<20} {trad_n_groups:<20} {trad_n_outliers:<20}")
    
    # Calculăm statisticile pentru abordarea de învățare profundă
    deep_n_groups = len([g for g in deep_groups.keys() if not str(g).startswith('outlier_')])
    deep_n_outliers = len([g for g in deep_groups.keys() if str(g).startswith('outlier_')])
    print(f"{'Învățare profundă':<20} {deep_n_groups:<20} {deep_n_outliers:<20}")
    
    # Calculăm statisticile pentru abordarea ensemble
    ensemble_n_groups = len([g for g in ensemble_groups.keys() if not str(g).startswith('outlier_')])
    ensemble_n_outliers = len([g for g in ensemble_groups.keys() if str(g).startswith('outlier_')])
    print(f"{'Ensemble':<20} {ensemble_n_groups:<20} {ensemble_n_outliers:<20}")
    
    # Creăm un grafic pentru a compara rezultatele
    plt.figure(figsize=(12, 8))
    
    # Definim datele pentru grafic
    approaches = ['Tradițională', 'Învățare profundă', 'Ensemble']
    n_groups = [trad_n_groups, deep_n_groups, ensemble_n_groups]
    n_outliers = [trad_n_outliers, deep_n_outliers, ensemble_n_outliers]
    
    # Creăm graficul
    x = np.arange(len(approaches))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width/2, n_groups, width, label='Număr de grupuri')
    rects2 = ax.bar(x + width/2, n_outliers, width, label='Număr de outlier-i')
    
    # Adăugăm etichetele și titlul
    ax.set_xlabel('Abordare')
    ax.set_ylabel('Număr')
    ax.set_title('Compararea rezultatelor pentru diferite abordări')
    ax.set_xticks(x)
    ax.set_xticklabels(approaches)
    ax.legend()
    
    # Adăugăm valorile pe bare
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(rects1)
    autolabel(rects2)
    
    fig.tight_layout()
    plt.savefig('results/comparison.png')
    logger.info("Graficul de comparare a fost salvat în results/comparison.png")
    
    # Creăm o matrice de confuzie pentru a compara grupurile
    print("\nMatrice de confuzie pentru grupurile comune:")
    
    # Obținem toate grupurile unice
    all_groups = set()
    for groups in [traditional_groups, deep_groups, ensemble_groups]:
        for group_id in groups.keys():
            if not str(group_id).startswith('outlier_'):
                all_groups.add(str(group_id))
    
    # Creăm un dicționar pentru a număra site-urile web comune între grupuri
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    # Populăm matricea de confuzie
    for website in common_websites:
        trad_group = str(traditional_labels[website])
        deep_group = str(deep_labels[website])
        ensemble_group = str(ensemble_labels[website])
        
        # Ignorăm outlier-ii
        if not trad_group.startswith('outlier_') and not deep_group.startswith('outlier_'):
            confusion_matrix[f"Trad_{trad_group}"][f"Deep_{deep_group}"] += 1
        
        if not trad_group.startswith('outlier_') and not ensemble_group.startswith('outlier_'):
            confusion_matrix[f"Trad_{trad_group}"][f"Ens_{ensemble_group}"] += 1
        
        if not deep_group.startswith('outlier_') and not ensemble_group.startswith('outlier_'):
            confusion_matrix[f"Deep_{deep_group}"][f"Ens_{ensemble_group}"] += 1
    
    # Afișăm matricea de confuzie
    print("\nMatrice de confuzie între abordarea tradițională și cea de învățare profundă:")
    for trad_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Trad_')]):
        for deep_group in sorted([g for g in confusion_matrix[trad_group].keys() if g.startswith('Deep_')]):
            if confusion_matrix[trad_group][deep_group] > 0:
                print(f"{trad_group} - {deep_group}: {confusion_matrix[trad_group][deep_group]} site-uri web comune")
    
    print("\nMatrice de confuzie între abordarea tradițională și cea ensemble:")
    for trad_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Trad_')]):
        for ensemble_group in sorted([g for g in confusion_matrix[trad_group].keys() if g.startswith('Ens_')]):
            if confusion_matrix[trad_group][ensemble_group] > 0:
                print(f"{trad_group} - {ensemble_group}: {confusion_matrix[trad_group][ensemble_group]} site-uri web comune")
    
    print("\nMatrice de confuzie între abordarea de învățare profundă și cea ensemble:")
    for deep_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Deep_')]):
        for ensemble_group in sorted([g for g in confusion_matrix[deep_group].keys() if g.startswith('Ens_')]):
            if confusion_matrix[deep_group][ensemble_group] > 0:
                print(f"{deep_group} - {ensemble_group}: {confusion_matrix[deep_group][ensemble_group]} site-uri web comune")

def main():
    """Funcția principală"""
    # Rulăm toate abordările
    run_all_approaches('logos.snappy(1).parquet')

if __name__ == "__main__":
    main() 