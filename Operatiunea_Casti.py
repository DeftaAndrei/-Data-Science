from PIL import Image
import os
import numpy as np

def decode_pixels(pixels):
    """Încearcă să decodeze pixeli în text"""
    try:
        text = ''
        for pixel in pixels:
            if isinstance(pixel, tuple):
                for value in pixel:
                    if 32 <= value <= 126:  # Doar caracterele ASCII imprimabile
                        text += chr(value)
            else:
                if 32 <= pixel <= 126:
                    text += chr(pixel)
        return text
    except:
        return ""

def find_veridion_message(image_path):
    try:
        img = Image.open(image_path)
        img_array = np.array(img)
        
        print(f"\n{'='*50}")
        print(f"Analiză pentru {os.path.basename(image_path)}:")
        print(f"{'='*50}")
        
        # Verificăm toți pixelii pentru text ascuns
        pixels = list(img.getdata())
        decoded_text = decode_pixels(pixels)
        
        # Căutăm cuvântul "veridion" în textul decodat
        if "veridion" in decoded_text.lower():
            print("\nMesaj găsit care conține 'veridion':")
            # Încercăm să extragem contextul în jurul cuvântului "veridion"
            start_idx = decoded_text.lower().find("veridion")
            context_start = max(0, start_idx - 20)
            context_end = min(len(decoded_text), start_idx + 20)
            print(decoded_text[context_start:context_end])
        
        # Verificăm canalele de culoare pentru mesaje ascunse
        if img.mode == 'RGBA':
            for i, channel in enumerate(['R', 'G', 'B', 'A']):
                channel_data = img_array[:,:,i]
                unique_values = np.unique(channel_data)
                
                # Verificăm dacă există pattern-uri în canal
                if len(unique_values) < 20:  # Mărim pragul pentru a găsi mai multe pattern-uri
                    channel_text = decode_pixels(unique_values)
                    if "veridion" in channel_text.lower():
                        print(f"\nMesaj găsit în canalul {channel}:")
                        print(channel_text)
        
        # Verificăm zonele transparente pentru mesaje ascunse
        if img.mode == 'RGBA':
            alpha_channel = img_array[:,:,3]
            transparent_pixels = np.sum(alpha_channel < 255)
            if transparent_pixels > 0:
                # Extragem pixelii transparenți
                transparent_data = img_array[alpha_channel < 255]
                transparent_text = decode_pixels(transparent_data)
                if "veridion" in transparent_text.lower():
                    print("\nMesaj găsit în zonele transparente:")
                    print(transparent_text)
        
        # Verificăm pattern-uri în dimensiunile imaginii
        width, height = img.size
        if width % 8 == 0 and height % 8 == 0:
            # Încercăm să extragem informații din dimensiuni
            dimension_text = decode_pixels([width, height])
            if "veridion" in dimension_text.lower():
                print("\nMesaj găsit în dimensiunile imaginii:")
                print(dimension_text)
                
    except Exception as e:
        print(f"Eroare la analizarea {image_path}: {str(e)}")

def main():
    images = [
        'new_thumbs_up_emoji.png',
        'new_spongbob.png',
        'new_skibidi_veridion.png',
        'new_yummy_microplastics.png',
        'new_filthy_veridion.png'
    ]
    
    print("Începem căutarea mesajelor secrete care conțin 'veridion' în imagini...")
    print("Voi analiza fiecare imagine pentru a găsi mesajul specific asociat cu 'veridion'")
    print("\n")
    
    for image in images:
        if os.path.exists(image):
            find_veridion_message(image)
        else:
            print(f"Fișierul {image} nu a fost găsit.")

if __name__ == "__main__":
    main()
