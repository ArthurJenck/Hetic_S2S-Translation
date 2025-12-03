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
import time
from nltk.translate.bleu_score import sentence_bleu


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
    
    def build_attention_mechanism(self, encoder_outputs, decoder_outputs):
        """
        Implémente le mécanisme d'attention dot-product.
        
        Args:
            encoder_outputs: Sorties de l'encodeur (batch, encoder_timesteps, latent_dim)
            decoder_outputs: Sorties du décodeur (batch, decoder_timesteps, latent_dim)
        
        Returns:
            context_vector: Vecteur de contexte pondéré
            attention_weights: Poids d'attention (pour visualisation)
        """
        # Étape 1 : Calcul des scores d'attention (similarité)
        attention_scores = Dot(axes=[2, 2], name='attention_scores')([decoder_outputs, encoder_outputs])
        
        # Étape 2 : Normalisation avec softmax
        attention_weights = Softmax(name='attention_weights')(attention_scores)
        
        # Étape 3 : Vecteur de contexte (moyenne pondérée)
        context_vector = Dot(axes=[2, 1], name='context_vector')([attention_weights, encoder_outputs])
        
        print(f"\n=== MÉCANISME D'ATTENTION ===")
        print(f"Scores: Dot([decoder, encoder]) → shape {attention_scores.shape}")
        print(f"Weights: Softmax(scores) → shape {attention_weights.shape}")
        print(f"Context: Dot([weights, encoder]) → shape {context_vector.shape}")
        
        return context_vector, attention_weights
    
    def build_decoder(self):
        """
        Construit le décodeur : Input → Embedding → LSTM → Attention → Concatenate → Dense.
        """
        self.decoder_inputs = Input(shape=(None,), name='decoder_input')
        
        dec_emb = Embedding(self.vocab_size_en, self.embedding_dim, 
                           mask_zero=True, name='decoder_embedding')(self.decoder_inputs)
        
        self.decoder_lstm = LSTM(self.latent_dim, return_sequences=True, 
                                return_state=True, name='decoder_lstm')
        decoder_outputs, _, _ = self.decoder_lstm(dec_emb, initial_state=self.encoder_states)
        
        # Attention
        context_vector, self.attention_weights = self.build_attention_mechanism(
            self.encoder_outputs, decoder_outputs)
        
        # Concaténation du contexte et de la sortie LSTM
        decoder_combined = Concatenate(axis=-1, name='concat')([context_vector, decoder_outputs])
        
        # Couche Dense pour prédiction
        self.decoder_dense = Dense(self.vocab_size_en, activation='softmax', name='decoder_output')
        self.decoder_outputs = self.decoder_dense(decoder_combined)
        
        print(f"\n=== DÉCODEUR ===")
        print(f"Input: shape=(None,)")
        print(f"Embedding: {self.vocab_size_en} mots → {self.embedding_dim} dim")
        print(f"LSTM: {self.latent_dim} unités (initial_state=encoder_states)")
        print(f"Attention + Concatenate")
        print(f"Dense: {self.vocab_size_en} sorties (softmax)")
        print(f"decoder_outputs: {self.decoder_outputs.shape}")
    
    def build_model(self):
        """Construit le modèle Seq2Seq complet en assemblant encodeur et décodeur."""
        self.build_encoder()
        self.build_decoder()
        
        self.model = Model([self.encoder_inputs, self.decoder_inputs], 
                          self.decoder_outputs, 
                          name='seq2seq_training')
        
        print(f"\n=== MODÈLE SEQ2SEQ COMPLET ===")
        self.model.summary()
        
        return self.model
    
    def compile_model(self):
        """Compile le modèle avec optimizer, loss et metrics."""
        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print(f"\n=== COMPILATION ===")
        print(f"Optimizer: adam")
        print(f"Loss: sparse_categorical_crossentropy")
        print(f"Metrics: accuracy")
    
    def setup_callbacks(self):
        """Configure les callbacks pour l'entraînement."""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                verbose=1
            )
        ]
        
        print(f"\n=== CALLBACKS ===")
        print(f"✓ EarlyStopping (patience=5)")
        print(f"✓ ModelCheckpoint (best_model.keras)")
        print(f"✓ ReduceLROnPlateau (factor=0.5, patience=3)")
        
        return callbacks
    
    def train(self, batch_size: int = 64, validation_split: float = 0.2):
        """
        Entraîne le modèle Seq2Seq.
        
        Args:
            batch_size: Taille des batchs
            validation_split: Proportion pour validation
        """
        self.compile_model()
        callbacks = self.setup_callbacks()
        
        print(f"\n=== ENTRAÎNEMENT ===")
        print(f"Batch size: {batch_size}")
        print(f"Epochs: {self.epochs}")
        print(f"Validation split: {validation_split * 100}%")
        print(f"Nombre d'exemples entraînement: {int(len(self.encoder_input_data) * (1 - validation_split))}")
        print(f"Nombre d'exemples validation: {int(len(self.encoder_input_data) * validation_split)}")
        print()
        
        self.history = self.model.fit(
            [self.encoder_input_data, self.decoder_input_data],
            self.decoder_target_data,
            batch_size=batch_size,
            epochs=self.epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        print(f"\n✓ Entraînement terminé!")
        print(f"Meilleure val_loss: {min(self.history.history['val_loss']):.4f}")
        
        return self.history
    
    def build_inference_models(self):
        """Construit les modèles séparés pour l'inférence (sans teacher forcing)."""
        # Encodeur d'inférence
        self.encoder_model = Model(
            self.encoder_inputs,
            [self.encoder_outputs, self.state_h, self.state_c],
            name='encoder_inference'
        )
        
        # Décodeur d'inférence - Inputs séparés
        decoder_inputs_inf = Input(shape=(None,), name='decoder_input_inf')
        encoder_outputs_inf = Input(shape=(None, self.latent_dim), name='encoder_outputs_inf')
        decoder_state_input_h = Input(shape=(self.latent_dim,), name='decoder_state_h_inf')
        decoder_state_input_c = Input(shape=(self.latent_dim,), name='decoder_state_c_inf')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        
        # Embedding
        dec_emb_inf = Embedding(self.vocab_size_en, self.embedding_dim, 
                               mask_zero=True, name='decoder_embedding_inf')(decoder_inputs_inf)
        
        # LSTM
        decoder_outputs_inf, state_h_inf, state_c_inf = self.decoder_lstm(
            dec_emb_inf, initial_state=decoder_states_inputs
        )
        decoder_states_inf = [state_h_inf, state_c_inf]
        
        # Attention
        attention_scores_inf = Dot(axes=[2, 2])([decoder_outputs_inf, encoder_outputs_inf])
        attention_weights_inf = Softmax()(attention_scores_inf)
        context_vector_inf = Dot(axes=[2, 1])([attention_weights_inf, encoder_outputs_inf])
        
        # Concatenate + Dense
        decoder_combined_inf = Concatenate(axis=-1)([context_vector_inf, decoder_outputs_inf])
        decoder_outputs_final = self.decoder_dense(decoder_combined_inf)
        
        # Modèle décodeur d'inférence
        self.decoder_model = Model(
            [decoder_inputs_inf, encoder_outputs_inf, decoder_state_input_h, decoder_state_input_c],
            [decoder_outputs_final, state_h_inf, state_c_inf],
            name='decoder_inference'
        )
        
        print(f"\n=== MODÈLES D'INFÉRENCE ===")
        print(f"✓ encoder_model: Input → [encoder_outputs, state_h, state_c]")
        print(f"✓ decoder_model: [decoder_input, encoder_outputs, h, c] → [output, new_h, new_c]")
    
    def decode_sequence(self, input_seq):
        """
        Décode une séquence d'entrée en utilisant greedy decoding.
        
        Args:
            input_seq: Séquence source encodée (numpy array)
        
        Returns:
            Phrase traduite (string)
        """
        # Étape 1 : Encoder la phrase source
        enc_outputs, h, c = self.encoder_model.predict(input_seq, verbose=0)
        
        # Étape 2 : Initialiser avec <sos>
        sos_token = self.tokenizer_en.word_index['<sos>']
        target_seq = np.array([[sos_token]])
        
        # Étape 3 : Boucle de génération
        decoded_sentence = []
        stop_condition = False
        
        while not stop_condition:
            # Prédire le prochain mot
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq, enc_outputs, h, c],
                verbose=0
            )
            
            # Greedy decoding : prendre le mot le plus probable
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            
            # Convertir l'index en mot
            sampled_word = self.tokenizer_en.index_word.get(sampled_token_index, '')
            
            # Ajouter le mot si ce n'est pas <eos>
            if sampled_word and sampled_word != '<eos>':
                decoded_sentence.append(sampled_word)
            
            # Condition d'arrêt
            if sampled_word == '<eos>' or len(decoded_sentence) >= self.max_len_en:
                stop_condition = True
            
            # Préparer l'entrée pour le prochain timestep
            target_seq = np.array([[sampled_token_index]])
        
        return ' '.join(decoded_sentence)
    
    def translate(self, french_sentence: str) -> str:
        """
        Traduit une phrase française en anglais.
        
        Args:
            french_sentence: Phrase en français
        
        Returns:
            Traduction en anglais
        """
        # Prétraiter
        french_sentence = french_sentence.lower().strip()
        
        # Tokeniser
        seq = self.tokenizer_fr.texts_to_sequences([french_sentence])
        
        # Pad
        input_seq = pad_sequences(seq, maxlen=self.max_len_fr, padding='post')
        
        # Décoder
        translation = self.decode_sequence(input_seq)
        
        return translation
    
    def evaluate_exact_match(self, n_samples: int = 50) -> float:
        """
        Évalue l'exact match accuracy sur n_samples.
        
        Args:
            n_samples: Nombre d'exemples à tester
        
        Returns:
            Accuracy en pourcentage
        """
        n_samples = min(n_samples, len(self.input_texts))
        correct = 0
        
        print(f"\n=== EXACT MATCH ACCURACY ({n_samples} exemples) ===")
        
        for i in range(n_samples):
            # Traduction
            prediction = self.translate(self.input_texts[i])
            true_translation = self.target_texts[i].lower().strip()
            
            # Comparaison
            if prediction == true_translation:
                correct += 1
        
        accuracy = (correct / n_samples) * 100
        print(f"Exact Match: {correct}/{n_samples} = {accuracy:.2f}%")
        
        return accuracy
    
    def evaluate_bleu(self, n_samples: int = 100):
        """
        Calcule le score BLEU moyen sur n_samples.
        
        Args:
            n_samples: Nombre d'exemples à tester
        
        Returns:
            Score BLEU moyen
        """
        n_samples = min(n_samples, len(self.input_texts))
        bleu_scores = []
        times = []
        
        print(f"\n=== ÉVALUATION BLEU ({n_samples} exemples) ===")
        
        for i in range(n_samples):
            start = time.time()
            
            # Traduction
            prediction = self.translate(self.input_texts[i])
            
            elapsed = time.time() - start
            times.append(elapsed)
            
            # Référence (liste de listes pour BLEU)
            reference = [self.target_texts[i].lower().split()]
            candidate = prediction.split()
            
            # Score BLEU
            try:
                score = sentence_bleu(reference, candidate)
                bleu_scores.append(score)
            except:
                bleu_scores.append(0.0)
        
        mean_bleu = np.mean(bleu_scores)
        std_bleu = np.std(bleu_scores)
        mean_time = np.mean(times)
        
        print(f"BLEU moyen: {mean_bleu:.4f} (±{std_bleu:.4f})")
        print(f"Temps moyen par traduction: {mean_time*1000:.2f}ms")
        
        return mean_bleu
    
    def show_examples(self, n_samples: int = 20):
        """
        Affiche des exemples de traductions.
        
        Args:
            n_samples: Nombre d'exemples à afficher
        """
        n_samples = min(n_samples, len(self.input_texts))
        
        print(f"\n=== EXEMPLES DE TRADUCTIONS ({n_samples}) ===\n")
        
        good_examples = []
        bad_examples = []
        
        for i in range(len(self.input_texts)):
            if len(good_examples) >= n_samples // 2 and len(bad_examples) >= n_samples // 2:
                break
            
            prediction = self.translate(self.input_texts[i])
            true_translation = self.target_texts[i].lower().strip()
            
            is_correct = (prediction == true_translation)
            
            example = {
                'fr': self.input_texts[i],
                'pred': prediction,
                'true': true_translation,
                'correct': is_correct
            }
            
            if is_correct and len(good_examples) < n_samples // 2:
                good_examples.append(example)
            elif not is_correct and len(bad_examples) < n_samples // 2:
                bad_examples.append(example)
        
        print("✓ BONNES TRADUCTIONS:")
        for i, ex in enumerate(good_examples, 1):
            print(f"{i}. FR: {ex['fr']}")
            print(f"   EN: {ex['pred']}")
            print()
        
        print("\n✗ MAUVAISES TRADUCTIONS:")
        for i, ex in enumerate(bad_examples, 1):
            print(f"{i}. FR: {ex['fr']}")
            print(f"   Prédiction: {ex['pred']}")
            print(f"   Vérité:     {ex['true']}")
            print()
    
    def plot_training_history(self):
        """Visualise les courbes de loss et accuracy durant l'entraînement."""
        if self.history is None:
            print("Aucun historique d'entraînement disponible.")
            return
        
        print(f"\n=== VISUALISATION DE L'ENTRAÎNEMENT ===")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Loss
        ax1.plot(self.history.history['loss'], label='Loss (train)', linewidth=2, color='#3498db')
        ax1.plot(self.history.history['val_loss'], label='Loss (validation)', linewidth=2, color='#e74c3c')
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Évolution de la Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Accuracy
        ax2.plot(self.history.history['accuracy'], label='Accuracy (train)', linewidth=2, color='#2ecc71')
        ax2.plot(self.history.history['val_accuracy'], label='Accuracy (validation)', linewidth=2, color='#f39c12')
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Évolution de l\'Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
        print("✓ Graphiques sauvegardés dans 'training_history.png'")
        plt.show()

