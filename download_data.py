import zipfile
import os


def download_dataset():
    zip_path = 'fra-eng.zip'
    
    if not os.path.exists('fra.txt'):
        print("Téléchargement du dataset fra-eng.zip...")
        
        try:
            import requests
            url = 'http://www.manythings.org/anki/fra-eng.zip'
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            with open(zip_path, 'wb') as f:
                f.write(response.content)
            print("✓ Téléchargement terminé")
        except ImportError:
            print("Module 'requests' manquant. Installez-le : pip install requests")
            return
        except Exception as e:
            print(f"Erreur: {e}")
            print("Téléchargez manuellement depuis: http://www.manythings.org/anki/")
            return
        
        print("Extraction...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('fra.txt')
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
        print("✓ Dataset prêt")
    else:
        print("✓ Dataset déjà présent")


if __name__ == "__main__":
    download_dataset()

