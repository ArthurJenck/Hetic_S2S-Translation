from translator import Translator
import os


if __name__ == "__main__":
    if not os.path.exists('fra.txt'):
        print("Dataset manquant. Exécutez d'abord: python download_data.py")
        exit(1)
    
    translator = Translator(max_samples=500, epochs=10)
    
    translator.load_data(num_samples=500)
    translator.build_tokenizers()
    translator.tokenize_sequences()
    translator.prepare_training_data()
    
    translator.build_encoder()

