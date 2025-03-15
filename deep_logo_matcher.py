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

# Configurare logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DeepLogoMatcher:
    def __init__(self, parquet_file):
        """
        Inițializează clasa DeepLogoMatcher
        
        Args:
            parquet_file (str): Calea către fișierul parquet cu datele despre logourile site-urilor
        """
        self.parquet_file = parquet_file
        self.data = None
        self.features = None
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
            self.features = self._extract_features_from_binary()
        elif 'logo_url' in self.data.columns:
            # Dacă avem URL-uri către logourile companiilor
            self.features = self._extract_features_from_urls()
        elif 'logo_base64' in self.data.columns:
            # Dacă avem logourile codificate în base64
            self.features = self._extract_features_from_base64()
        elif 'logo' in self.data.columns:
            # Verificăm tipul de date pentru coloana 'logo'
            if self.data['logo'].dtype == 'object':
                first_val = self.data['logo'].iloc[0] if len(self.data) > 0 else None
                if isinstance(first_val, bytes):
                    self.features = self._extract_features_from_binary('logo')
                elif isinstance(first_val, str):
                    if first_val.startswith('http'):
                        self.features = self._extract_features_from_urls('logo')
                    else:
                        self.features = self._extract_features_from_base64('logo')
        else:
            logger.error("Nu s-a găsit nicio coloană cu date despre logourile companiilor.")
            return False
        
        logger.info(f"Caracteristici extrase pentru {len(self.features)} logouri.")
        return True
    
    def _extract_features_from_binary(self, column='logo_binary'):
        """Extrage caracteristicile din datele binare ale logourilor"""
        features = []
        
        def process_binary(idx, binary_data):
            try:
                # Convertim datele binare în imagine
                img = Image.open(io.BytesIO(binary_data))
                # Extragem caracteristicile
                return self._extract_deep_features(img)
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului pentru indexul {idx}: {str(e)}")
                return None
        
        # Folosim ThreadPoolExecutor pentru a procesa logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            binary_data_list = self.data[column].tolist()
            results = list(executor.map(process_binary, range(len(binary_data_list)), binary_data_list))
        
        return results
    
    def _extract_features_from_urls(self, column='logo_url'):
        """Extrage caracteristicile din URL-urile logourilor"""
        features = []
        
        def process_url(idx, url):
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    img = Image.open(BytesIO(response.content))
                    return self._extract_deep_features(img)
                else:
                    logger.warning(f"Nu s-a putut descărca logoul de la URL-ul: {url}")
                    return None
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului de la URL-ul {url}: {str(e)}")
                return None
        
        # Folosim ThreadPoolExecutor pentru a descărca logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            urls = self.data[column].tolist()
            results = list(executor.map(process_url, range(len(urls)), urls))
        
        return results
    
    def _extract_features_from_base64(self, column='logo_base64'):
        """Extrage caracteristicile din logourile codificate în base64"""
        features = []
        
        def process_base64(idx, base64_data):
            try:
                # Decodificăm datele base64
                img_data = base64.b64decode(base64_data)
                img = Image.open(io.BytesIO(img_data))
                # Extragem caracteristicile
                return self._extract_deep_features(img)
            except Exception as e:
                logger.warning(f"Eroare la procesarea logoului pentru indexul {idx}: {str(e)}")
                return None
        
        # Folosim ThreadPoolExecutor pentru a procesa logourile în paralel
        with ThreadPoolExecutor(max_workers=10) as executor:
            base64_data_list = self.data[column].tolist()
            results = list(executor.map(process_base64, range(len(base64_data_list)), base64_data_list))
        
        return results
    
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
    
    def cluster_logos(self, eps=0.3, min_samples=2):
        """
        Grupează logourile similare folosind algoritmul DBSCAN
        
        Args:
            eps (float): Parametrul epsilon pentru DBSCAN
            min_samples (int): Numărul minim de eșantioane pentru DBSCAN
        """
        if self.features is None:
            logger.error("Nu există caracteristici extrase. Rulați mai întâi metoda preprocess_logos().")
            return False
        
        # Eliminăm caracteristicile None
        valid_indices = [i for i, f in enumerate(self.features) if f is not None]
        valid_features = [self.features[i] for i in valid_indices]
        
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
    
    def visualize_clusters(self, output_file='deep_clusters_visualization.png'):
        """
        Vizualizează grupurile de logourile folosind t-SNE
        
        Args:
            output_file (str): Calea către fișierul de ieșire pentru vizualizare
        """
        if self.features is None or self.website_groups is None:
            logger.error("Nu există caracteristici sau grupuri. Rulați mai întâi metodele preprocess_logos() și cluster_logos().")
            return False
        
        # Eliminăm caracteristicile None
        valid_indices = [i for i, f in enumerate(self.features) if f is not None]
        valid_features = [self.features[i] for i in valid_indices]
        
        if not valid_features:
            logger.error("Nu există caracteristici valide pentru vizualizare.")
            return False
        
        # Convertim lista de caracteristici într-un array NumPy
        X = np.array(valid_features)
        
        # Aplicăm t-SNE pentru a reduce dimensionalitatea la 2D
        from sklearn.manifold import TSNE
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
        
        plt.title('Grupuri de logourile similare (t-SNE)')
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
        
        print("\n=== GRUPURI DE SITE-URI WEB CU LOGOURILE SIMILARE (DEEP LEARNING) ===\n")
        for group_id, websites in self.website_groups.items():
            if str(group_id).startswith('outlier_'):
                print(f"Grup unic (outlier): {websites[0]}")
            else:
                print(f"Grup {group_id} ({len(websites)} site-uri):")
                for website in websites:
                    print(f"  - {website}")
            print()
        
        return True

def main():
    """Funcția principală"""
    # Inițializăm clasa DeepLogoMatcher
    matcher = DeepLogoMatcher('logos.snappy(1).parquet')
    
    # Încărcăm datele
    if not matcher.load_data():
        return
    
    # Preprocesăm logourile și extragem caracteristicile
    if not matcher.preprocess_logos():
        return
    
    # Grupăm logourile similare
    if not matcher.cluster_logos(eps=0.3, min_samples=2):
        return
    
    # Afișăm grupurile de site-uri web
    matcher.print_website_groups()
    
    # Vizualizăm grupurile
    matcher.visualize_clusters()

if __name__ == "__main__":
    main() 