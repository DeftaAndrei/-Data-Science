import pandas as pd
import numpy as np
import sys
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt

def inspect_parquet(file_path):
    """
    Inspectează structura fișierului parquet și afișează informații despre date
    
    Args:
        file_path (str): Calea către fișierul parquet
    """
    try:
        # Încărcăm datele
        print(f"Încărcăm datele din {file_path}...")
        df = pd.read_parquet(file_path)
        
        # Afișăm informații generale
        print(f"\nInformații generale:")
        print(f"Număr de înregistrări: {len(df)}")
        print(f"Coloane disponibile: {df.columns.tolist()}")
        
        # Afișăm primele rânduri
        print(f"\nPrimele 5 rânduri:")
        print(df.head())
        
        # Afișăm informații despre tipurile de date
        print(f"\nTipuri de date:")
        print(df.dtypes)
        
        # Verificăm dacă există coloane pentru logourile companiilor
        logo_columns = [col for col in df.columns if 'logo' in col.lower()]
        if logo_columns:
            print(f"\nColoane cu logourile companiilor: {logo_columns}")
            
            # Verificăm tipul de date pentru fiecare coloană cu logourile
            for col in logo_columns:
                print(f"\nTipul de date pentru {col}: {df[col].dtype}")
                
                # Verificăm primele valori
                print(f"Primele valori pentru {col}:")
                for i, val in enumerate(df[col].head()):
                    if isinstance(val, bytes):
                        print(f"  {i}: Bytes de lungime {len(val)}")
                    elif isinstance(val, str):
                        if val.startswith('http'):
                            print(f"  {i}: URL - {val}")
                        elif len(val) > 100:
                            print(f"  {i}: String de lungime {len(val)}")
                        else:
                            print(f"  {i}: {val}")
                    else:
                        print(f"  {i}: {type(val)} - {val}")
                
                # Încercăm să afișăm primul logo
                try:
                    if df[col].dtype == 'object' and len(df) > 0:
                        first_val = df[col].iloc[0]
                        
                        if isinstance(first_val, bytes):
                            # Încercăm să deschidem datele binare ca imagine
                            img = Image.open(io.BytesIO(first_val))
                            plt.figure(figsize=(5, 5))
                            plt.imshow(img)
                            plt.title(f"Primul logo din {col}")
                            plt.axis('off')
                            plt.savefig(f"first_logo_{col}.png")
                            print(f"Primul logo a fost salvat în first_logo_{col}.png")
                        
                        elif isinstance(first_val, str):
                            if first_val.startswith('http'):
                                print(f"Prima valoare este un URL: {first_val}")
                            elif len(first_val) > 100:
                                # Încercăm să decodificăm ca base64
                                try:
                                    img_data = base64.b64decode(first_val)
                                    img = Image.open(io.BytesIO(img_data))
                                    plt.figure(figsize=(5, 5))
                                    plt.imshow(img)
                                    plt.title(f"Primul logo din {col} (decodificat din base64)")
                                    plt.axis('off')
                                    plt.savefig(f"first_logo_{col}_base64.png")
                                    print(f"Primul logo (decodificat din base64) a fost salvat în first_logo_{col}_base64.png")
                                except Exception as e:
                                    print(f"Nu s-a putut decodifica valoarea ca base64: {str(e)}")
                except Exception as e:
                    print(f"Nu s-a putut afișa primul logo: {str(e)}")
        
        # Verificăm dacă există o coloană pentru site-urile web
        website_columns = [col for col in df.columns if 'website' in col.lower() or 'url' in col.lower() or 'site' in col.lower()]
        if website_columns:
            print(f"\nColoane cu site-urile web: {website_columns}")
            
            # Afișăm primele valori
            for col in website_columns:
                print(f"\nPrimele valori pentru {col}:")
                print(df[col].head())
        
        return df
    
    except Exception as e:
        print(f"Eroare la inspectarea fișierului parquet: {str(e)}")
        return None

if __name__ == "__main__":
    # Verificăm dacă a fost specificat un fișier
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        file_path = 'logos.snappy(1).parquet'
    
    # Inspectăm fișierul
    df = inspect_parquet(file_path) 