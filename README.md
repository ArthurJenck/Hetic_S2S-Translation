# Hetic_S2S-Translation

Modèle neuronal de traduction (anglais vers français) en Seq2Seq avec LSTM, mécanisme d'attention et teacher forcing.

## Installation

Installer les dépendances :

```bash
pip install -r requirements.txt
```

## Utilisation

1. Télécharger le dataset :

```bash
python download_data.py
```

2. Lancer l'entraînement et la démo :

```bash
python main.py
```

Le script entraîne le modèle, affiche des exemples de traduction, évalue les performances (BLEU score) et lance une interface de traduction interactive dans la console.
