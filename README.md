# Number Plate Recognition (Pandas + Keras/JAX)

Reconnaissance de plaques d'immatriculation par réseau de neurones convolutif.

## Pipeline

1. **Prepare** — Extraire des caractères étiquetés à partir de photos de plaques.
2. **Train** — Entraîner un CNN sur EMNIST digits, sur des images extraites, ou les deux.
3. **Infer** — Lire le texte d'une plaque à partir d'une image.

## Structure du projet

```text
plate_recognition/
├── main.py                     # Point d'entrée CLI (prepare / train / infer)
├── requirements.txt
├── emnist-digits-train.csv.zip # Dataset EMNIST digits
├── emnist-digits-test.csv.zip
├── images_treatment.py         # Prototype initial (corner detection)
├── src/
│   ├── model.py                # Définition du CNN (Keras)
│   ├── train_emnist.py         # Entraînement (EMNIST + dossier d'images)
│   ├── prepare_data.py         # Extraction de caractères depuis des photos
│   ├── infer_plate.py          # Inférence sur images de plaques
│   └── utils_image.py          # Détection de plaque, correction de perspective, segmentation
├── models/
│   ├── digit_model.keras       # Modèle entraîné
│   └── label_map.json          # Mapping caractère → index
└── data/
    └── chars/                  # Caractères extraits (généré par prepare)
```

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export KERAS_BACKEND=jax
```

## Utilisation

### 1. Préparer des données depuis des photos de plaques

Nommer chaque image avec le texte de la plaque (ex: `AB123CD.png`, `AB-123-CD.jpg`).

```bash
python main.py prepare --input path/to/plate_photos --output data/chars
```

Options :
- `--assume-cropped-plate` — si les images sont déjà recadrées sur la plaque
- `--no-perspective` — désactiver la correction de perspective

### 2. Entraîner le modèle

**Avec EMNIST (digits uniquement) :**
```bash
python main.py train --epochs 5 --batch-size 128
```

**Avec des images extraites (lettres + chiffres) :**
```bash
python main.py train --train-dir data/chars --epochs 10
```

**Combinaison EMNIST + images (merge) :**
```bash
python main.py train --train-dir data/chars --merge --epochs 10
```

**Smoke test rapide :**
```bash
python main.py train --epochs 1 --train-limit 5000 --test-limit 1000 --batch-size 64
```

### 3. Inférence

```bash
python main.py infer --image path/to/plate.jpg
python main.py infer --folder path/to/images
python main.py infer --folder path/to/images --assume-cropped-plate
python main.py infer --folder path/to/images --debug-out outputs/debug
```

## Notes

- Le modèle reconnaît 36 classes (0-9, A-Z) quand il est entraîné avec des images de plaques.
- La détection de plaque est heuristique ; utiliser `--assume-cropped-plate` si instable.
- La correction de perspective améliore les résultats sur les photos prises en angle.
- Le fichier `label_map.json` est généré automatiquement pendant l'entraînement.
