import urllib.request
import zipfile
import os


def download_dataset():
    url = 'http://www.manythings.org/anki/fra-eng.zip'
    zip_path = 'fra-eng.zip'
    
    if not os.path.exists('fra.txt'):
        print("Téléchargement du dataset fra-eng.zip...")
        urllib.request.urlretrieve(url, zip_path)
        print("✓ Téléchargement terminé")
        
        print("Extraction de fra.txt...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extract('fra.txt')
        print("✓ Extraction terminée")
        
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("✓ Fichier zip supprimé")
    else:
        print("✓ Le fichier fra.txt existe déjà")


if __name__ == "__main__":
    download_dataset()

