import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, Dense, Dot, Softmax, Concatenate
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional


class Translator:
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256, 
                 latent_dim: int = 512, max_samples: Optional[int] = None, epochs: int = 10):
        """
        Initialise le traducteur Seq2Seq avec attention.
        
        Args:
            vocab_size: Taille max du vocabulaire par langue
            embedding_dim: Dimension des embeddings
            latent_dim: Dimension de l'espace latent des LSTM
            max_samples: Nombre max d'exemples (None = tous)
            epochs: Nombre d'époques d'entraînement
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.max_samples = max_samples
        self.epochs = epochs
        
        self.input_texts = []
        self.target_texts = []
        
        self.tokenizer_fr = None
        self.tokenizer_en = None
        self.vocab_size_fr = None
        self.vocab_size_en = None
        
        self.max_len_fr = None
        self.max_len_en = None
        
        self.encoder_input_data = None
        self.decoder_input_data = None
        self.decoder_target_data = None
        
        print(f"TensorFlow version: {tf.__version__}")
    
    def load_data(self, filepath: str = 'fra.txt', num_samples: Optional[int] = None) -> Tuple[List[str], List[str]]:
        """
        Charge les paires français-anglais depuis le fichier TSV.
        
        Args:
            filepath: Chemin vers fra.txt
            num_samples: Limite d'exemples (None = tous)
        
        Returns:
            Listes des phrases françaises et anglaises
        """
        input_texts = []
        target_texts = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')
        
        if num_samples is not None:
            lines = lines[:min(num_samples, len(lines))]
        
        for line in lines:
            if '\t' not in line:
                continue
            
            parts = line.split('\t')
            if len(parts) >= 2:
                eng_text = parts[0]
                fr_text = parts[1]
                
                eng_text = eng_text.lower().strip()
                fr_text = fr_text.lower().strip()
                
                input_texts.append(fr_text)
                target_texts.append(eng_text)
        
        self.input_texts = input_texts
        self.target_texts = target_texts
        
        print(f"\n=== DONNÉES CHARGÉES ===")
        print(f"Nombre de paires: {len(input_texts)}")
        print(f"Exemples:")
        for i in range(min(3, len(input_texts))):
            print(f"  FR: {input_texts[i]}")
            print(f"  EN: {target_texts[i]}")
        
        return input_texts, target_texts
    
    def build_tokenizers(self):
        """Construit les tokenizers français et anglais avec vocabulaire limité."""
        self.tokenizer_fr = Tokenizer(num_words=self.vocab_size, 
                                       oov_token='<unk>',
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        self.tokenizer_en = Tokenizer(num_words=self.vocab_size,
                                       oov_token='<unk>',
                                       filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        
        self.tokenizer_fr.fit_on_texts(self.input_texts)
        self.tokenizer_en.fit_on_texts(self.target_texts)
        
        self.tokenizer_en.word_index['<sos>'] = len(self.tokenizer_en.word_index) + 1
        self.tokenizer_en.word_index['<eos>'] = len(self.tokenizer_en.word_index) + 1
        self.tokenizer_en.index_word[self.tokenizer_en.word_index['<sos>']] = '<sos>'
        self.tokenizer_en.index_word[self.tokenizer_en.word_index['<eos>']] = '<eos>'
        
        self.vocab_size_fr = min(len(self.tokenizer_fr.word_index) + 1, self.vocab_size)
        self.vocab_size_en = len(self.tokenizer_en.word_index) + 1
        
        print(f"\n=== VOCABULAIRES ===")
        print(f"Vocabulaire français: {self.vocab_size_fr} mots")
        print(f"Vocabulaire anglais: {self.vocab_size_en} mots")
        print(f"Token <sos>: index {self.tokenizer_en.word_index['<sos>']}")
        print(f"Token <eos>: index {self.tokenizer_en.word_index['<eos>']}")
    
    def tokenize_sequences(self):
        """Convertit les phrases en séquences d'indices avec tokens <sos> et <eos>."""
        self.encoder_input_seqs = self.tokenizer_fr.texts_to_sequences(self.input_texts)
        
        target_seqs = self.tokenizer_en.texts_to_sequences(self.target_texts)
        
        sos_token = self.tokenizer_en.word_index['<sos>']
        eos_token = self.tokenizer_en.word_index['<eos>']
        
        self.decoder_target_seqs = []
        for seq in target_seqs:
            self.decoder_target_seqs.append([sos_token] + seq + [eos_token])
        
        self.max_len_fr = max(len(seq) for seq in self.encoder_input_seqs)
        self.max_len_en = max(len(seq) for seq in self.decoder_target_seqs)
        
        print(f"\n=== SÉQUENCES TOKENISÉES ===")
        print(f"Longueur max français: {self.max_len_fr}")
        print(f"Longueur max anglais: {self.max_len_en}")
    
    def prepare_training_data(self):
        """Prépare les données avec padding et teacher forcing."""
        self.encoder_input_data = pad_sequences(self.encoder_input_seqs, 
                                                 maxlen=self.max_len_fr, 
                                                 padding='post')
        
        decoder_target_padded = pad_sequences(self.decoder_target_seqs,
                                               maxlen=self.max_len_en,
                                               padding='post')
        
        decoder_input_seqs = [seq[:-1] for seq in self.decoder_target_seqs]
        self.decoder_input_data = pad_sequences(decoder_input_seqs,
                                                 maxlen=self.max_len_en - 1,
                                                 padding='post')
        
        self.decoder_target_data = np.array([seq[1:] for seq in decoder_target_padded])
        
        print(f"\n=== DONNÉES D'ENTRAÎNEMENT ===")
        print(f"encoder_input_data: {self.encoder_input_data.shape}")
        print(f"decoder_input_data: {self.decoder_input_data.shape}")
        print(f"decoder_target_data: {self.decoder_target_data.shape}")
    
    def build_encoder(self):
        """Construit l'encodeur : Input → Embedding → LSTM."""
        self.encoder_inputs = Input(shape=(None,), name='encoder_input')
        
        enc_emb = Embedding(self.vocab_size_fr, self.embedding_dim, 
                           mask_zero=True, name='encoder_embedding')(self.encoder_inputs)
        
        encoder_lstm = LSTM(self.latent_dim, return_sequences=True, 
                           return_state=True, name='encoder_lstm')
        self.encoder_outputs, self.state_h, self.state_c = encoder_lstm(enc_emb)
        self.encoder_states = [self.state_h, self.state_c]
        
        print(f"\n=== ENCODEUR ===")
        print(f"Input: shape=(None,)")
        print(f"Embedding: {self.vocab_size_fr} mots → {self.embedding_dim} dim")
        print(f"LSTM: {self.latent_dim} unités")
        print(f"encoder_outputs: {self.encoder_outputs.shape}")
        print(f"state_h: {self.state_h.shape}")
        print(f"state_c: {self.state_c.shape}")

