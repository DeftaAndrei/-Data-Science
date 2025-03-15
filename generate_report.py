import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import time
import os
from sklearn.metrics import adjusted_rand_score
from collections import defaultdict
import json
from datetime import datetime

# Importăm clasele din celelalte scripturi
from logo_matcher import LogoMatcher
from deep_logo_matcher import DeepLogoMatcher
from ensemble_logo_matcher import EnsembleLogoMatcher

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_report(parquet_file, eps=0.3, min_samples=2, output_dir='results'):
    """
    Generează un raport detaliat despre rezultatele potrivirii logourilor
    
    Args:
        parquet_file (str): Calea către fișierul parquet cu datele despre logourile site-urilor
        eps (float): Parametrul epsilon pentru DBSCAN
        min_samples (int): Numărul minim de eșantioane pentru DBSCAN
        output_dir (str): Directorul pentru rezultate
    """
    # Creăm directorul pentru rezultate
    os.makedirs(output_dir, exist_ok=True)
    
    # Creăm un dicționar pentru a stoca rezultatele
    report = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {
            'eps': eps,
            'min_samples': min_samples
        },
        'approaches': {},
        'comparison': {}
    }
    
    # Rulăm abordarea tradițională
    logger.info("=== RULĂM ABORDAREA TRADIȚIONALĂ ===")
    start_time = time.time()
    traditional_matcher = LogoMatcher(parquet_file)
    traditional_matcher.load_data()
    traditional_matcher.preprocess_logos()
    traditional_matcher.cluster_logos(eps=eps, min_samples=min_samples)
    traditional_matcher.print_website_groups()
    traditional_matcher.visualize_clusters(output_file=f'{output_dir}/traditional_clusters.png')
    traditional_time = time.time() - start_time
    logger.info(f"Abordarea tradițională a durat {traditional_time:.2f} secunde.")
    
    # Adăugăm rezultatele abordării tradiționale în raport
    traditional_groups = traditional_matcher.get_website_groups()
    report['approaches']['traditional'] = {
        'execution_time': traditional_time,
        'n_groups': len([g for g in traditional_groups.keys() if not str(g).startswith('outlier_')]),
        'n_outliers': len([g for g in traditional_groups.keys() if str(g).startswith('outlier_')]),
        'groups': {str(k): v for k, v in traditional_groups.items()}
    }
    
    # Rulăm abordarea de învățare profundă
    logger.info("\n=== RULĂM ABORDAREA DE ÎNVĂȚARE PROFUNDĂ ===")
    start_time = time.time()
    deep_matcher = DeepLogoMatcher(parquet_file)
    deep_matcher.load_data()
    deep_matcher.preprocess_logos()
    deep_matcher.cluster_logos(eps=eps, min_samples=min_samples)
    deep_matcher.print_website_groups()
    deep_matcher.visualize_clusters(output_file=f'{output_dir}/deep_clusters.png')
    deep_time = time.time() - start_time
    logger.info(f"Abordarea de învățare profundă a durat {deep_time:.2f} secunde.")
    
    # Adăugăm rezultatele abordării de învățare profundă în raport
    deep_groups = deep_matcher.get_website_groups()
    report['approaches']['deep_learning'] = {
        'execution_time': deep_time,
        'n_groups': len([g for g in deep_groups.keys() if not str(g).startswith('outlier_')]),
        'n_outliers': len([g for g in deep_groups.keys() if str(g).startswith('outlier_')]),
        'groups': {str(k): v for k, v in deep_groups.items()}
    }
    
    # Rulăm abordarea ensemble
    logger.info("\n=== RULĂM ABORDAREA ENSEMBLE ===")
    start_time = time.time()
    ensemble_matcher = EnsembleLogoMatcher(parquet_file)
    ensemble_matcher.load_data()
    ensemble_matcher.preprocess_logos()
    results = ensemble_matcher.evaluate_clustering()
    ensemble_matcher.print_website_groups()
    ensemble_matcher.visualize_clusters(output_file=f'{output_dir}/ensemble_clusters.png')
    ensemble_time = time.time() - start_time
    logger.info(f"Abordarea ensemble a durat {ensemble_time:.2f} secunde.")
    
    # Adăugăm rezultatele abordării ensemble în raport
    ensemble_groups = ensemble_matcher.get_website_groups()
    report['approaches']['ensemble'] = {
        'execution_time': ensemble_time,
        'n_groups': len([g for g in ensemble_groups.keys() if not str(g).startswith('outlier_')]),
        'n_outliers': len([g for g in ensemble_groups.keys() if str(g).startswith('outlier_')]),
        'groups': {str(k): v for k, v in ensemble_groups.items()},
        'evaluation_results': [
            {
                'eps': result['eps'],
                'min_samples': result['min_samples'],
                'n_clusters': result['n_clusters'],
                'n_outliers': result['n_outliers'],
                'silhouette_score': result['silhouette_score']
            }
            for result in results
        ] if results else []
    }
    
    # Comparăm rezultatele
    comparison_results = compare_results(traditional_matcher, deep_matcher, ensemble_matcher, output_dir)
    report['comparison'] = comparison_results
    
    # Salvăm raportul în format JSON
    with open(f'{output_dir}/report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # Generăm raportul în format HTML
    generate_html_report(report, output_dir)
    
    logger.info(f"Raportul a fost generat și salvat în {output_dir}/report.json și {output_dir}/report.html")
    
    return report

def compare_results(traditional_matcher, deep_matcher, ensemble_matcher, output_dir):
    """
    Compară rezultatele obținute cu diferite abordări
    
    Args:
        traditional_matcher (LogoMatcher): Instanța clasei LogoMatcher
        deep_matcher (DeepLogoMatcher): Instanța clasei DeepLogoMatcher
        ensemble_matcher (EnsembleLogoMatcher): Instanța clasei EnsembleLogoMatcher
        output_dir (str): Directorul pentru rezultate
    
    Returns:
        dict: Rezultatele comparației
    """
    # Obținem grupurile de site-uri web
    traditional_groups = traditional_matcher.get_website_groups()
    deep_groups = deep_matcher.get_website_groups()
    ensemble_groups = ensemble_matcher.get_website_groups()
    
    # Verificăm dacă avem grupuri valide
    if traditional_groups is None or deep_groups is None or ensemble_groups is None:
        logger.error("Nu există grupuri valide pentru comparație.")
        return {}
    
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
        return {}
    
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
    plt.savefig(f'{output_dir}/comparison.png')
    logger.info(f"Graficul de comparare a fost salvat în {output_dir}/comparison.png")
    
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
    trad_deep_matrix = []
    for trad_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Trad_')]):
        for deep_group in sorted([g for g in confusion_matrix[trad_group].keys() if g.startswith('Deep_')]):
            if confusion_matrix[trad_group][deep_group] > 0:
                print(f"{trad_group} - {deep_group}: {confusion_matrix[trad_group][deep_group]} site-uri web comune")
                trad_deep_matrix.append({
                    'trad_group': trad_group,
                    'deep_group': deep_group,
                    'count': confusion_matrix[trad_group][deep_group]
                })
    
    print("\nMatrice de confuzie între abordarea tradițională și cea ensemble:")
    trad_ensemble_matrix = []
    for trad_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Trad_')]):
        for ensemble_group in sorted([g for g in confusion_matrix[trad_group].keys() if g.startswith('Ens_')]):
            if confusion_matrix[trad_group][ensemble_group] > 0:
                print(f"{trad_group} - {ensemble_group}: {confusion_matrix[trad_group][ensemble_group]} site-uri web comune")
                trad_ensemble_matrix.append({
                    'trad_group': trad_group,
                    'ensemble_group': ensemble_group,
                    'count': confusion_matrix[trad_group][ensemble_group]
                })
    
    print("\nMatrice de confuzie între abordarea de învățare profundă și cea ensemble:")
    deep_ensemble_matrix = []
    for deep_group in sorted([g for g in confusion_matrix.keys() if g.startswith('Deep_')]):
        for ensemble_group in sorted([g for g in confusion_matrix[deep_group].keys() if g.startswith('Ens_')]):
            if confusion_matrix[deep_group][ensemble_group] > 0:
                print(f"{deep_group} - {ensemble_group}: {confusion_matrix[deep_group][ensemble_group]} site-uri web comune")
                deep_ensemble_matrix.append({
                    'deep_group': deep_group,
                    'ensemble_group': ensemble_group,
                    'count': confusion_matrix[deep_group][ensemble_group]
                })
    
    # Creăm un dicționar cu rezultatele comparației
    comparison_results = {
        'common_websites': len(common_websites),
        'adjusted_rand_scores': {
            'traditional_vs_deep': trad_deep_score,
            'traditional_vs_ensemble': trad_ensemble_score,
            'deep_vs_ensemble': deep_ensemble_score
        },
        'statistics': {
            'traditional': {
                'n_groups': trad_n_groups,
                'n_outliers': trad_n_outliers
            },
            'deep_learning': {
                'n_groups': deep_n_groups,
                'n_outliers': deep_n_outliers
            },
            'ensemble': {
                'n_groups': ensemble_n_groups,
                'n_outliers': ensemble_n_outliers
            }
        },
        'confusion_matrices': {
            'traditional_vs_deep': trad_deep_matrix,
            'traditional_vs_ensemble': trad_ensemble_matrix,
            'deep_vs_ensemble': deep_ensemble_matrix
        }
    }
    
    return comparison_results

def generate_html_report(report, output_dir):
    """
    Generează un raport în format HTML
    
    Args:
        report (dict): Dicționarul cu rezultatele
        output_dir (str): Directorul pentru rezultate
    """
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Raport de potrivire a logourilor</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
                line-height: 1.6;
            }}
            h1, h2, h3 {{
                color: #333;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin-bottom: 20px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            .container {{
                display: flex;
                flex-wrap: wrap;
                justify-content: space-between;
            }}
            .image-container {{
                width: 30%;
                margin-bottom: 20px;
            }}
            .image-container img {{
                width: 100%;
            }}
            .group {{
                margin-bottom: 10px;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
            }}
            .group-title {{
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .website {{
                margin-left: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Raport de potrivire a logourilor</h1>
        <p>Generat la: {report['timestamp']}</p>
        
        <h2>Parametri</h2>
        <table>
            <tr>
                <th>Parametru</th>
                <th>Valoare</th>
            </tr>
            <tr>
                <td>eps</td>
                <td>{report['parameters']['eps']}</td>
            </tr>
            <tr>
                <td>min_samples</td>
                <td>{report['parameters']['min_samples']}</td>
            </tr>
        </table>
        
        <h2>Rezultate pentru fiecare abordare</h2>
        <table>
            <tr>
                <th>Abordare</th>
                <th>Timp de execuție (s)</th>
                <th>Număr de grupuri</th>
                <th>Număr de outlier-i</th>
            </tr>
    """
    
    for approach, data in report['approaches'].items():
        html += f"""
            <tr>
                <td>{approach.replace('_', ' ').title()}</td>
                <td>{data['execution_time']:.2f}</td>
                <td>{data['n_groups']}</td>
                <td>{data['n_outliers']}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h2>Compararea rezultatelor</h2>
        <h3>Scorul Rand ajustat</h3>
        <table>
            <tr>
                <th>Comparație</th>
                <th>Scor</th>
            </tr>
    """
    
    for comparison, score in report['comparison']['adjusted_rand_scores'].items():
        html += f"""
            <tr>
                <td>{comparison.replace('_', ' vs ').title()}</td>
                <td>{score:.4f}</td>
            </tr>
        """
    
    html += """
        </table>
        
        <h3>Vizualizări</h3>
        <div class="container">
            <div class="image-container">
                <h4>Abordarea tradițională</h4>
                <img src="traditional_clusters.png" alt="Grupuri tradiționale">
            </div>
            <div class="image-container">
                <h4>Abordarea de învățare profundă</h4>
                <img src="deep_clusters.png" alt="Grupuri de învățare profundă">
            </div>
            <div class="image-container">
                <h4>Abordarea ensemble</h4>
                <img src="ensemble_clusters.png" alt="Grupuri ensemble">
            </div>
        </div>
        
        <div class="image-container" style="width: 100%;">
            <h4>Compararea rezultatelor</h4>
            <img src="comparison.png" alt="Compararea rezultatelor">
        </div>
        
        <h2>Grupuri de site-uri web</h2>
    """
    
    for approach, data in report['approaches'].items():
        html += f"""
        <h3>Abordarea {approach.replace('_', ' ').title()}</h3>
        <div class="groups-container">
        """
        
        for group_id, websites in data['groups'].items():
            if not group_id.startswith('outlier_'):
                html += f"""
                <div class="group">
                    <div class="group-title">Grup {group_id} ({len(websites)} site-uri)</div>
                """
                
                for website in websites:
                    html += f"""
                    <div class="website">{website}</div>
                    """
                
                html += """
                </div>
                """
        
        html += """
        </div>
        
        <h4>Outlier-i</h4>
        <div class="groups-container">
        """
        
        for group_id, websites in data['groups'].items():
            if group_id.startswith('outlier_'):
                html += f"""
                <div class="group">
                    <div class="group-title">Outlier: {websites[0]}</div>
                </div>
                """
        
        html += """
        </div>
        """
    
    if 'ensemble' in report['approaches'] and 'evaluation_results' in report['approaches']['ensemble'] and report['approaches']['ensemble']['evaluation_results']:
        html += """
        <h2>Evaluarea parametrilor pentru abordarea ensemble</h2>
        <table>
            <tr>
                <th>eps</th>
                <th>min_samples</th>
                <th>Număr de clustere</th>
                <th>Număr de outlier-i</th>
                <th>Scor silhouette</th>
            </tr>
        """
        
        for result in report['approaches']['ensemble']['evaluation_results']:
            html += f"""
            <tr>
                <td>{result['eps']}</td>
                <td>{result['min_samples']}</td>
                <td>{result['n_clusters']}</td>
                <td>{result['n_outliers']}</td>
                <td>{result['silhouette_score']:.4f}</td>
            </tr>
            """
        
        html += """
        </table>
        """
    
    html += """
    </body>
    </html>
    """
    
    # Salvăm raportul HTML
    with open(f'{output_dir}/report.html', 'w') as f:
        f.write(html)

def main():
    """Funcția principală"""
    # Generăm raportul
    generate_report('logos.snappy(1).parquet')

if __name__ == "__main__":
    main() 