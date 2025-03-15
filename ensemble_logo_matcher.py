import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from concurrent.futures import ThreadPoolExecutor
import requests
from io import BytesIO
import time
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import adjusted_rand_score

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnsembleLogoMatcher:
    def __init__(self, parquet_file):
        """
        Inițializează clasa EnsembleLogoMatcher
        
        Args:
            parquet_file (str): Calea către fișierul parquet cu datele despre logourile site-urilor
        """
        self.parquet_file = parquet_file
        self.data = None
        self.traditional_features = None
        self.deep_features = None
        self.combined_features = None
        self.website_groups = None
        
        # Inițializăm modelul de extragere a caracteristicilor
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Utilizăm dispozitivul: {self.device}")
        
        # Încărcăm modelul ResNet pre-antrenat
        self.model = models.resnet50(pretrained=True)
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])  # Eliminăm ultimul strat
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Definim transformările pentru preprocesarea imaginilor
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_data(self):
        """Încarcă datele din fișierul parquet"""
        try:
            self.data = pd.read_parquet(self.parquet_file)
            logger.info(f"Date încărcate cu succes. Număr de înregistrări: {len(self.data)}")
            logger.info(f"Coloane disponibile: {self.data.columns.tolist()}")
            return True
        except Exception as e:
            logger.error(f"Eroare la încărcarea datelor: {str(e)}")
            return False
    
    def preprocess_logos(self):
        """Preprocesează logourile și extrage caracteristicile"""
        if self.data is None:
            logger.error("Nu există date încărcate. Rulați mai întâi metoda load_data().")
            return False
        
        # Verificăm coloanele disponibile și adaptăm codul în funcție de acestea
        if 'logo_binary' in self.data.columns:
            # Dacă avem logourile stocate ca date binare
            self.traditional_features, self.deep_features = self._extract_features_from_binary()
        elif 'logo_url' in self.data.columns:
            # Dacă avem URL-uri către logourile companiilor
            self.traditional_features, self.deep_features = self._extract_features_from_urls()
        elif 'logo_base64' in self.data.columns:
            # Dacă avem logourile codificate în base64
            self.traditional_features, self.deep_features = self._extract_features_from_base64()
        elif 'logo' in self.data.columns:
            # Verificăm tipul de date pentru coloana 'logo'
            if self.data['logo'].dtype == 'object':
                first_val = self.data['logo'].iloc[0] if len(self.data) > 0 else None
                if isinstance(first_val, bytes):
                    self.traditional_features, self.deep_features = self._extract_features_from_binary('logo')
                elif isinstance(first_val, str):
                    if first_val.startswith('http'):
                        self.traditional_features, self.deep_features = self._extract_features_from_urls('logo')
                    else:
                        self.traditional_features, self.deep_features = self._extract_features_from_base64('logo')
        else:
            logger.error("Nu s-a găsit nicio coloană cu date despre logourile companiilor.")
            return False
        
        # Combinăm caracteristicile tradiționale și cele de învățare profundă
        self._combine_features()
        
        logger.info(f"Caracteristici extrase pentru {len(self.combined_features)} logouri.")
        return True
    
    def _extract_features_from_binary(self, column='logo_binary'):
        """Extrage caracteristicile din datele binare ale logourilor"""
        traditional_features = []
        deep_features = []
        
        def process_binary(idx, binary_data):
            try:
                # Convertim datele binare în imagine
                img = Image.open(io.BytesIO(binary_data))
                # Extragem caracteristicile
                trad_feat = self._extract_traditional_features(img)
                deep_feat = self._extract_deep_features(img)
                return trad_feat, deep_feat
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului pentru indexul {idx}: {str(e)}")
                return None, None
        
        # Folosim ThreadPoolExecutor pentru a procesa logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            binary_data_list = self.data[column].tolist()
            results = list(executor.map(process_binary, range(len(binary_data_list)), binary_data_list))
        
        # Separăm rezultatele
        for trad_feat, deep_feat in results:
            traditional_features.append(trad_feat)
            deep_features.append(deep_feat)
        
        return traditional_features, deep_features
    
    def _extract_features_from_urls(self, column='logo_url'):
        """Extrage caracteristicile din URL-urile logourilor"""
        traditional_features = []
        deep_features = []
        
        def process_url(idx, url):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    trad_feat = self._extract_traditional_features(img)
                    deep_feat = self._extract_deep_features(img)
                    return trad_feat, deep_feat
                else:
                    logger.warning(f"Nu s-a putut descărca logoul de la URL-ul: {url}")
                    return None, None
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului de la URL-ul {url}: {str(e)}")
                return None, None
        
        # Folosim ThreadPoolExecutor pentru a descărca logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            urls = self.data[column].tolist()
            results = list(executor.map(process_url, range(len(urls)), urls))
        
        # Separăm rezultatele
        for trad_feat, deep_feat in results:
            traditional_features.append(trad_feat)
            deep_features.append(deep_feat)
        
        return traditional_features, deep_features
    
    def _extract_features_from_base64(self, column='logo_base64'):
        """Extrage caracteristicile din logourile codificate în base64"""
        traditional_features = []
        deep_features = []
        
        def process_base64(idx, base64_data):
            try:
                # Decodificăm datele base64
                img_data = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(img_data))
                trad_feat = self._extract_traditional_features(img)
                deep_feat = self._extract_deep_features(img)
                return trad_feat, deep_feat
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului pentru indexul {idx}: {str(e)}")
                return None, None
        
        # Folosim ThreadPoolExecutor pentru a procesa logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            base64_data_list = self.data[column].tolist()
            results = list(executor.map(process_base64, range(len(base64_data_list)), base64_data_list))
        
        # Separăm rezultatele
        for trad_feat, deep_feat in results:
            traditional_features.append(trad_feat)
            deep_features.append(deep_feat)
        
        return traditional_features, deep_features
    
    def _extract_traditional_features(self, img):
        """
        Extrage caracteristicile tradiționale din imagine
        
        Args:
            img (PIL.Image): Imaginea logoului
            
        Returns:
            np.ndarray: Vector de caracteristici
        """
        try:
            # Convertim imaginea la formatul OpenCV
            img_cv = np.array(img.convert('RGB'))
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
            
            # Redimensionăm imaginea la o dimensiune fixă
            img_resized = cv2.resize(img_cv, (100, 100))
            
            # Extragem histograma de culori
            hist_b = cv2.calcHist([img_resized], [0], None, [8], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_resized], [1], None, [8], [0, 256]).flatten()
            hist_r = cv2.calcHist([img_resized], [2], None, [8], [0, 256]).flatten()
            
            # Normalizăm histogramele
            hist_b = hist_b / np.sum(hist_b) if np.sum(hist_b) > 0 else hist_b
            hist_g = hist_g / np.sum(hist_g) if np.sum(hist_g) > 0 else hist_g
            hist_r = hist_r / np.sum(hist_r) if np.sum(hist_r) > 0 else hist_r
            
            # Combinăm histogramele
            color_features = np.concatenate([hist_b, hist_g, hist_r])
            
            # Extragem caracteristici de formă și textură
            # Convertim la grayscale
            img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
            
            # Aplicăm detectorul de margini Canny
            edges = cv2.Canny(img_gray, 100, 200)
            
            # Calculăm histograma de orientare a gradientului (HOG simplificat)
            gx = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=1)
            gy = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=1)
            mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
            
            # Calculăm histograma de orientare
            bins = 9
            hist_hog = np.zeros(bins)
            for i in range(img_gray.shape[0]):
                for j in range(img_gray.shape[1]):
                    if edges[i, j] > 0:  # Considerăm doar pixelii de margine
                        bin_idx = int(angle[i, j] / (180.0 / bins)) % bins
                        hist_hog[bin_idx] += mag[i, j]
            
            # Normalizăm histograma HOG
            hist_hog = hist_hog / np.sum(hist_hog) if np.sum(hist_hog) > 0 else hist_hog
            
            # Combinăm toate caracteristicile
            features = np.concatenate([color_features, hist_hog])
            
            return features
        except Exception as e:
            logger.warning(f"Eroare la extragerea caracteristicilor tradiționale: {str(e)}")
            return None
    
    def _extract_deep_features(self, img):
        """
        Extrage caracteristicile din imagine folosind un model de învățare profundă
        
        Args:
            img (PIL.Image): Imaginea logoului
            
        Returns:
            np.ndarray: Vector de caracteristici
        """
        try:
            # Preprocesăm imaginea
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            
            # Extragem caracteristicile
            with torch.no_grad():
                features = self.model(img_tensor)
            
            # Convertim la numpy array și aplatizăm
            features = features.squeeze().cpu().numpy()
            
            return features
        except Exception as e:
            logger.warning(f"Eroare la extragerea caracteristicilor profunde: {str(e)}")
            return None
    
    def _combine_features(self):
        """Combină caracteristicile tradiționale și cele de învățare profundă"""
        self.combined_features = []
        
        # Verificăm dacă avem caracteristici valide
        if self.traditional_features is None or self.deep_features is None:
            logger.error("Nu există caracteristici extrase.")
            return False
        
        # Combinăm caracteristicile
        for i in range(len(self.traditional_features)):
            trad_feat = self.traditional_features[i]
            deep_feat = self.deep_features[i]
            
            if trad_feat is None or deep_feat is None:
                self.combined_features.append(None)
            else:
                # Normalizăm caracteristicile profunde
                deep_feat_norm = deep_feat / np.linalg.norm(deep_feat) if np.linalg.norm(deep_feat) > 0 else deep_feat
                
                # Redimensionăm caracteristicile profunde pentru a avea aceeași importanță ca cele tradiționale
                deep_feat_resized = cv2.resize(deep_feat_norm.reshape(1, -1), (trad_feat.shape[0], 1)).flatten()
                
                # Combinăm caracteristicile
                combined = np.concatenate([trad_feat, deep_feat_resized])
                self.combined_features.append(combined)
        
        logger.info(f"Caracteristicile au fost combinate cu succes.")
        return True
    
    def cluster_logos(self, eps=0.3, min_samples=2):
        """
        Grupează logourile similare folosind algoritmul DBSCAN
        
        Args:
            eps (float): Parametrul epsilon pentru DBSCAN
            min_samples (int): Numărul minim de eșantioane pentru DBSCAN
        """
        if self.combined_features is None:
            logger.error("Nu există caracteristici combinate. Rulați mai întâi metoda preprocess_logos().")
            return False
        
        # Eliminăm caracteristicile None
        valid_indices = [i for i, f in enumerate(self.combined_features) if f is not None]
        valid_features = [self.combined_features[i] for i in valid_indices]
        
        if not valid_features:
            logger.error("Nu există caracteristici valide pentru clustering.")
            return False
        
        # Convertim lista de caracteristici într-un array NumPy
        X = np.array(valid_features)
        
        # Calculăm matricea de similaritate cosinus
        similarity_matrix = cosine_similarity(X)
        
        # Convertim similaritatea în distanță (1 - similaritate)
        distance_matrix = 1 - similarity_matrix
        
        # Aplicăm algoritmul DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        
        # Creăm grupurile de site-uri web
        self.website_groups = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:  # Ignorăm outlier-ii (label = -1)
                idx = valid_indices[i]
                website = self.data.iloc[idx].get('website', f"Site {idx}")
                self.website_groups[label].append(website)
        
        # Adăugăm site-urile care nu au fost grupate (outlier-i) în grupuri separate
        outlier_indices = [valid_indices[i] for i, label in enumerate(labels) if label == -1]
        for i, idx in enumerate(outlier_indices):
            website = self.data.iloc[idx].get('website', f"Site {idx}")
            self.website_groups[f"outlier_{i}"] = [website]
        
        logger.info(f"Logourile au fost grupate în {len(self.website_groups)} grupuri.")
        return True
    
    def visualize_clusters(self, output_file='ensemble_clusters_visualization.png'):
        """
        Vizualizează grupurile de logourile folosind t-SNE
        
        Args:
            output_file (str): Calea către fișierul de ieșire pentru vizualizare
        """
        if self.combined_features is None or self.website_groups is None:
            logger.error("Nu există caracteristici sau grupuri. Rulați mai întâi metodele preprocess_logos() și cluster_logos().")
            return False
        
        # Eliminăm caracteristicile None
        valid_indices = [i for i, f in enumerate(self.combined_features) if f is not None]
        valid_features = [self.combined_features[i] for i in valid_indices]
        
        if not valid_features:
            logger.error("Nu există caracteristici valide pentru vizualizare.")
            return False
        
        # Convertim lista de caracteristici într-un array NumPy
        X = np.array(valid_features)
        
        # Aplicăm t-SNE pentru a reduce dimensionalitatea la 2D
        tsne = TSNE(n_components=2, random_state=42)
        X_tsne = tsne.fit_transform(X)
        
        # Creăm un dicționar care mapează fiecare index valid la eticheta grupului său
        idx_to_group = {}
        for group_id, websites in self.website_groups.items():
            for website in websites:
                for i, idx in enumerate(valid_indices):
                    if self.data.iloc[idx].get('website', f"Site {idx}") == website:
                        idx_to_group[i] = group_id
        
        # Creăm figura
        plt.figure(figsize=(12, 8))
        
        # Definim o paletă de culori
        colors = plt.cm.tab20(np.linspace(0, 1, len(self.website_groups)))
        color_map = {group_id: colors[i] for i, group_id in enumerate(self.website_groups.keys())}
        
        # Plotăm punctele
        for i in range(len(X_tsne)):
            group_id = idx_to_group.get(i, 'unknown')
            color = color_map.get(group_id, 'gray')
            marker = 'o' if not str(group_id).startswith('outlier_') else 'x'
            plt.scatter(X_tsne[i, 0], X_tsne[i, 1], c=[color], marker=marker, s=100, alpha=0.7)
            
            # Adăugăm eticheta site-ului web
            website = self.data.iloc[valid_indices[i]].get('website', f"Site {valid_indices[i]}")
            plt.annotate(website, (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)
        
        plt.title('Grupuri de logourile similare (Ensemble + t-SNE)')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Salvăm figura
        plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        logger.info(f"Vizualizarea a fost salvată în fișierul {output_file}")
        
        return True
    
    def get_website_groups(self):
        """Returnează grupurile de site-uri web"""
        return self.website_groups
    
    def print_website_groups(self):
        """Afișează grupurile de site-uri web"""
        if self.website_groups is None:
            logger.error("Nu există grupuri. Rulați mai întâi metoda cluster_logos().")
            return False
        
        print("\n=== GRUPURI DE SITE-URI WEB CU LOGOURILE SIMILARE (ENSEMBLE) ===\n")
        for group_id, websites in self.website_groups.items():
            if str(group_id).startswith('outlier_'):
                print(f"Grup unic (outlier): {websites[0]}")
            else:
                print(f"Grup {group_id} ({len(websites)} site-uri):")
                for website in websites:
                    print(f"  - {website}")
            print()
        
        return True
    
    def evaluate_clustering(self, eps_values=[0.1, 0.2, 0.3, 0.4, 0.5], min_samples_values=[2, 3, 4, 5]):
        """
        Evaluează performanța clusteringului pentru diferite valori ale parametrilor
        
        Args:
            eps_values (list): Lista de valori pentru parametrul epsilon
            min_samples_values (list): Lista de valori pentru parametrul min_samples
        """
        if self.combined_features is None:
            logger.error("Nu există caracteristici combinate. Rulați mai întâi metoda preprocess_logos().")
            return False
        
        # Eliminăm caracteristicile None
        valid_indices = [i for i, f in enumerate(self.combined_features) if f is not None]
        valid_features = [self.combined_features[i] for i in valid_indices]
        
        if not valid_features:
            logger.error("Nu există caracteristici valide pentru evaluare.")
            return False
        
        # Convertim lista de caracteristici într-un array NumPy
        X = np.array(valid_features)
        
        # Calculăm matricea de similaritate cosinus
        similarity_matrix = cosine_similarity(X)
        
        # Convertim similaritatea în distanță (1 - similaritate)
        distance_matrix = 1 - similarity_matrix
        
        # Evaluăm pentru diferite valori ale parametrilor
        results = []
        best_score = -1
        best_params = None
        best_labels = None
        
        for eps in eps_values:
            for min_samples in min_samples_values:
                # Aplicăm algoritmul DBSCAN
                dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                labels = dbscan.fit_predict(distance_matrix)
                
                # Calculăm numărul de clustere (ignorând outlier-ii)
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                
                # Calculăm numărul de outlier-i
                n_outliers = list(labels).count(-1)
                
                # Calculăm scorul silhouette (dacă avem mai mult de un cluster)
                if n_clusters > 1:
                    from sklearn.metrics import silhouette_score
                    # Excludem outlier-ii pentru calculul scorului silhouette
                    mask = labels != -1
                    if np.sum(mask) > 1:
                        sil_score = silhouette_score(distance_matrix[mask][:, mask], labels[mask], metric='precomputed')
                    else:
                        sil_score = 0
                else:
                    sil_score = 0
                
                # Adăugăm rezultatele
                results.append({
                    'eps': eps,
                    'min_samples': min_samples,
                    'n_clusters': n_clusters,
                    'n_outliers': n_outliers,
                    'silhouette_score': sil_score
                })
                
                # Actualizăm cele mai bune rezultate
                if sil_score > best_score:
                    best_score = sil_score
                    best_params = (eps, min_samples)
                    best_labels = labels
        
        # Afișăm rezultatele
        print("\n=== EVALUAREA CLUSTERINGULUI ===\n")
        print("Rezultate pentru diferite valori ale parametrilor:")
        for result in results:
            print(f"eps={result['eps']}, min_samples={result['min_samples']}: {result['n_clusters']} clustere, {result['n_outliers']} outlier-i, silhouette_score={result['silhouette_score']:.4f}")
        
        print(f"\nCele mai bune rezultate: eps={best_params[0]}, min_samples={best_params[1]}, silhouette_score={best_score:.4f}")
        
        # Aplicăm clusteringul cu cei mai buni parametri
        if best_params is not None:
            self.cluster_logos(eps=best_params[0], min_samples=best_params[1])
        
        return results

def main():
    """Funcția principală"""
    # Inițializăm clasa EnsembleLogoMatcher
    matcher = EnsembleLogoMatcher('logos.snappy(1).parquet')
    
    # Încărcăm datele
    if not matcher.load_data():
        return
    
    # Preprocesăm logourile și extragem caracteristicile
    if not matcher.preprocess_logos():
        return
    
    # Evaluăm clusteringul pentru diferite valori ale parametrilor
    matcher.evaluate_clustering()
    
    # Afișăm grupurile de site-uri web
    matcher.print_website_groups()
    
    # Vizualizăm grupurile
    matcher.visualize_clusters()

if __name__ == "__main__":
    main() 