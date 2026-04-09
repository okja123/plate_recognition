# Guide de Présentation — Reconnaissance de Plaques d'Immatriculation

## Résumé du projet

Ce projet reconnaît le texte des plaques d'immatriculation européennes à partir de photos. Il utilise un réseau de neurones convolutif (CNN) entraîné avec Keras (backend JAX) sur des caractères extraits automatiquement depuis des photos réelles de plaques, combinés avec le dataset EMNIST digits.

Le pipeline complet fonctionne en 3 étapes : **Prepare → Train → Infer**.

---

## Architecture technique

### Stack technologique

| Composant | Technologie | Rôle |
|-----------|------------|------|
| Réseau de neurones | Keras 3.x + JAX | Construction et entraînement du CNN |
| Traitement d'images | OpenCV (cv2) | Détection de plaque, segmentation de caractères, correction de perspective |
| Données tabulaires | Pandas | Chargement du dataset EMNIST depuis des CSV zippés |
| Calcul numérique | NumPy | Manipulation des matrices d'images et tenseurs |
| Dataset de plaques | Kaggle (kagglehub) | 735 photos de plaques européennes annotées |
| Dataset de chiffres | EMNIST digits | 240 000 images de chiffres manuscrits/imprimés |

### Structure du CNN

```
Input (28x28x1 grayscale)
  → Conv2D(32, 3x3, relu, padding=same)
  → MaxPooling2D(2x2)
  → Conv2D(64, 3x3, relu, padding=same)
  → MaxPooling2D(2x2)
  → Flatten
  → Dense(128, relu)
  → Dropout(0.3)
  → Dense(36, softmax)    ← 36 classes : 0-9 + A-Z
```

Total : **424 996 paramètres** (1.62 MB)

### Pipeline de traitement d'image

```
Photo de plaque
  → Détection de la région de la plaque (contours + ratio largeur/hauteur)
  → Correction de perspective (transformation géométrique 4 points)
  → Binarisation adaptative (seuillage gaussien inversé)
  → Segmentation des caractères (contours + filtrage par taille/position)
  → Redimensionnement de chaque caractère sur canvas 28x28
  → Prédiction par le CNN
  → Texte de la plaque
```

### Structure des fichiers

```
plate_recognition/
├── main.py                      # Point d'entrée CLI (prepare / train / infer)
├── requirements.txt             # Dépendances Python
├── PRESENTATION.md              # Ce fichier
├── README.md                    # Documentation technique
├── Cahier_des_charge.txt        # Cahier des charges du projet
├── images_treatment.py          # Prototype initial (détection de coins)
│
├── src/
│   ├── model.py                 # Définition du CNN Keras
│   ├── train_emnist.py          # Entraînement (EMNIST + dossier d'images)
│   ├── prepare_data.py          # Extraction de caractères depuis photos de plaques
│   ├── infer_plate.py           # Inférence sur images de plaques
│   └── utils_image.py           # Détection de plaque, perspective, segmentation
│
├── models/
│   ├── digit_model.keras        # Modèle entraîné (sauvegardé après training)
│   └── label_map.json           # Mapping caractère → index (A=10, B=11, ..., Z=35)
│
├── data/
│   ├── chars_train/             # Caractères extraits pour entraînement (généré)
│   │   ├── 0/ 1/ 2/ ... 9/     # Sous-dossiers par chiffre
│   │   └── A/ B/ C/ ... Z/     # Sous-dossiers par lettre
│   └── chars_test/              # Caractères extraits pour test (généré)
│
├── datasets/                    # Dataset Kaggle (téléchargé)
│   └── abdelhamidzakaria/european-license-plates-dataset/versions/1/
│       └── dataset_final/
│           ├── train/           # 588 images de plaques
│           ├── test/            # 73 images
│           └── val/             # 74 images
│
├── emnist-digits-train.csv.zip  # EMNIST digits (240 000 échantillons)
└── emnist-digits-test.csv.zip   # EMNIST digits test (40 000 échantillons)
```

---

## Installation sur un nouveau laptop

### Prérequis

- Python 3.10 ou supérieur (testé avec 3.14)
- pip
- Compte Kaggle (pour télécharger le dataset de plaques)
- Environ 500 MB d'espace disque

### Étapes d'installation

```bash
# 1. Cloner ou copier le projet
cd ~/Documents
git clone <url_du_repo> plate_recognition
cd plate_recognition

# 2. Créer un environnement virtuel
python3 -m venv .venv
source .venv/bin/activate

# 3. Installer les dépendances
pip install pandas numpy keras "jax[cpu]" matplotlib opencv-python-headless

# 4. Configurer le backend Keras
export KERAS_BACKEND=jax

# 5. Télécharger le dataset Kaggle (nécessite un compte Kaggle)
pip install kagglehub
python3 -c "
import kagglehub, os
os.environ['KAGGLEHUB_CACHE'] = './'
path = kagglehub.dataset_download('abdelhamidzakaria/european-license-plates-dataset')
print('Downloaded to:', path)
"
```

**Note sur Kaggle** : lors du premier téléchargement, kagglehub demandera de se connecter. Suivre les instructions affichées dans le terminal (lien de connexion ou clé API à placer dans `~/.kaggle/kaggle.json`).

### Vérification rapide

```bash
# Vérifier que tout est installé
python3 -c "import keras, jax, cv2, pandas, numpy; print('Tout est installé !')"
```

---

## Commandes pour la démo live

**Important** : toujours activer le venv et exporter le backend avant chaque session :

```bash
cd plate_recognition
source .venv/bin/activate
export KERAS_BACKEND=jax
```

### Étape 1 — Préparer les données (extraire les caractères des photos)

```bash
# Extraire les caractères depuis les photos d'entraînement
python3 main.py prepare \
  --input datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/train \
  --output data/chars_train \
  --assume-cropped-plate \
  --no-perspective

# Extraire les caractères depuis les photos de test
python3 main.py prepare \
  --input datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/test \
  --output data/chars_test \
  --assume-cropped-plate \
  --no-perspective
```

**Ce que ça fait** : chaque photo de plaque est nommée avec le texte de la plaque (ex: `AB123CD.png`). Le script segmente chaque caractère individuellement, le redimensionne en 28x28 pixels, et le sauve dans un dossier par classe (`data/chars_train/A/`, `data/chars_train/B/`, etc.).

**Résultat attendu** : environ 1 684 images de caractères extraites en 36 classes (0-9, A-Z).

### Étape 2 — Entraîner le modèle

```bash
# Option A : entraîner uniquement sur les photos de plaques (rapide, ~2 secondes)
python3 main.py train \
  --train-dir data/chars_train \
  --test-dir data/chars_test \
  --epochs 10
```

**Résultat attendu** : ~94% de précision, entraînement en ~2 secondes.

```bash
# Option B : combiner avec EMNIST pour plus de données (recommandé, ~10 minutes)
python3 main.py train \
  --train-dir data/chars_train \
  --test-dir data/chars_test \
  --merge \
  --epochs 15
```

**Résultat attendu** : ~99.5% de précision, entraînement en ~10 minutes sur 241 684 échantillons.

### Étape 3 — Inférence (reconnaissance de plaque)

```bash
# Sur une seule image
python3 main.py infer \
  --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/SK560DP.png \
  --assume-cropped-plate

# Sur un dossier entier
python3 main.py infer \
  --folder datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val \
  --assume-cropped-plate

# Avec images de debug (montre les boîtes de segmentation)
python3 main.py infer \
  --folder datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val \
  --assume-cropped-plate \
  --debug-out outputs/debug
```

---

## Scénario de présentation suggéré

### Déroulé recommandé (15-20 minutes)

**1. Introduction (2 min)**
- Contexte : reconnaissance automatique de plaques d'immatriculation
- Applications : péages, parkings, contrôle routier, villes intelligentes
- Objectif du projet : construire un système complet from scratch avec Keras

**2. Architecture du pipeline (3 min)**
- Montrer le schéma : Photo → Détection → Perspective → Segmentation → CNN → Texte
- Expliquer chaque étape brièvement
- Insister sur le fait que le modèle est entraîné par nous, pas un modèle pré-entraîné

**3. Le dataset (2 min)**
- Montrer quelques photos de plaques du dataset Kaggle (ouvrir le dossier `datasets/.../train/`)
- Expliquer la convention de nommage : le nom du fichier = texte de la plaque
- 735 images de plaques européennes variées (Pays-Bas, France, Roumanie, Italie...)
- EMNIST : 240 000 images de chiffres pour renforcer la reconnaissance des chiffres

**4. Démo live — Préparation des données (3 min)**
- Lancer la commande `prepare` en live
- Montrer les images extraites dans `data/chars_train/` (ouvrir quelques dossiers A/, B/, 0/, 1/)
- Montrer qu'on obtient des images 28x28 en noir et blanc

**5. Démo live — Entraînement (3 min)**
- Lancer la commande `train` (option A pour rapidité en live)
- Montrer la progression des epochs : accuracy qui monte de 14% à 94%
- Expliquer : loss qui descend = le modèle apprend
- Mentionner que le modèle merged (option B) atteint 99.5%

**6. Démo live — Inférence (3 min)**
- Lancer sur une image isolée qui fonctionne bien, ex:
  ```bash
  python3 main.py infer --image datasets/.../val/SK560DP.png --assume-cropped-plate
  ```
  Résultat attendu : `SK560DP` ✓
- Lancer sur le dossier val complet pour montrer les résultats en masse
- Montrer les images debug si on a le temps (boîtes vertes autour des caractères)

**7. Résultats et limites (2 min)**
- Précision : 99.5% sur la classification de caractères individuels
- Limites connues :
  - La segmentation est heuristique (basée sur les contours), pas apprise
  - Plaques sombres, floues, ou à faible contraste posent problème
  - Certaines plaques européennes ont des formats très différents
- Améliorations possibles :
  - Utiliser un réseau de détection (YOLO, SSD) pour localiser les caractères
  - Augmentation de données (rotation, bruit, flou) pour rendre le modèle plus robuste
  - Ajouter les lettres EMNIST (pas juste digits) pour plus de données d'entraînement

**8. Transposition NSI lycée (2 min)** *(Phase 6 du projet)*
- Ce projet pourrait être adapté pour des élèves de terminale NSI
- Formation préalable nécessaire : introduction aux réseaux de neurones, notion de convolution
- TP possible : donner le dataset et le code de segmentation, faire entraîner un modèle simple
- Thème imposé ou libre ? Un thème imposé (reconnaissance de chiffres) est plus accessible
- Travail en binôme, sur 4-6 séances de 2h

### Images à montrer pendant la présentation

Pour le PowerPoint, voici les visuels utiles disponibles dans le projet :

1. **Photos de plaques** : ouvrir `datasets/.../train/` et montrer 4-5 plaques variées
2. **Caractères extraits** : ouvrir `data/chars_train/A/` et montrer les images 28x28
3. **Architecture du CNN** : le tableau du `model.summary()` (affiché pendant le training)
4. **Courbe d'entraînement** : les métriques accuracy/loss par epoch (copiées du terminal)
5. **Images de debug** : lancer avec `--debug-out` et montrer les boîtes de segmentation vertes
6. **Résultats d'inférence** : capture d'écran du terminal montrant les prédictions

---

## Exemples de plaques qui fonctionnent bien (pour la démo)

Tester ces images avant la présentation pour s'assurer qu'elles marchent :

```bash
# Ces plaques donnent de bons résultats
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/SK560DP.png
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/ER140CY.jpg
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/ZDT923_1.png
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/353ASK35.jpg
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/6923HJ2B.png
python3 main.py infer --assume-cropped-plate --image datasets/abdelhamidzakaria/european-license-plates-dataset/versions/1/dataset_final/val/G866969.png
```

---

## Résultats obtenus

### Entraînement sur photos de plaques uniquement (option A)

- Échantillons d'entraînement : 1 684
- Échantillons de test : 197
- Classes : 36 (0-9, A-Z)
- Epochs : 10
- Durée : ~2 secondes
- **Précision test : 94.4%**

### Entraînement combiné EMNIST + plaques (option B)

- Échantillons d'entraînement : 241 684
- Échantillons de test : 40 197
- Classes : 36 (0-9, A-Z)
- Epochs : 15
- Durée : ~10 minutes
- **Précision test : 99.5%**

### Progression de l'entraînement (option B)

```
Epoch  1/15 — accuracy: 96.3%  val_accuracy: 98.9%
Epoch  5/15 — accuracy: 99.4%  val_accuracy: 99.5%
Epoch 10/15 — accuracy: 99.7%  val_accuracy: 99.6%
Epoch 15/15 — accuracy: 99.8%  val_accuracy: 99.5%
```

---

## Problèmes courants et solutions

| Problème | Solution |
|----------|----------|
| `zsh: no matches found: jax[cpu]` | Mettre entre guillemets : `"jax[cpu]"` |
| `ModuleNotFoundError: No module named 'cv2'` | `pip install opencv-python-headless` |
| `ModuleNotFoundError: No module named 'keras'` | `pip install keras` + `export KERAS_BACKEND=jax` |
| `ModuleNotFoundError: No module named 'numpy'` | Vérifier que le venv est activé : `source .venv/bin/activate` |
| `scipy` ne s'installe pas | On n'en a pas besoin, il s'installe automatiquement avec jax |
| `zsh: command not found: python` | Utiliser `python3` au lieu de `python` |
| Kaggle demande une authentification | Aller sur kaggle.com → Settings → API → Create New Token, placer le fichier dans `~/.kaggle/kaggle.json` |
| "No characters segmented" sur une image | Normal pour certaines plaques difficiles, essayer avec `--no-perspective` ou sans `--assume-cropped-plate` |
| Le modèle n'existe pas | Lancer d'abord `python3 main.py train ...` |

---

## Choix techniques à justifier pendant l'oral

1. **Pourquoi Keras + JAX ?** — Keras est recommandé dans les consignes du cours. JAX est le backend le plus rapide sur CPU pour les petits modèles.

2. **Pourquoi un CNN et pas un MLP ?** — Les CNN exploitent la structure spatiale des images (convolutions). Un MLP traiterait chaque pixel indépendamment et perdrait l'information spatiale.

3. **Pourquoi EMNIST + photos de plaques ?** — EMNIST fournit beaucoup de données pour les chiffres (240K), les photos de plaques ajoutent les lettres et des caractères réalistes. La combinaison donne 99.5% de précision.

4. **Pourquoi la segmentation par contours ?** — Approche classique, sans besoin de modèle supplémentaire. C'est une limitation connue mais ça reste fidèle à l'esprit du projet (construire soi-même).

5. **Pourquoi 28x28 pixels ?** — Format standard MNIST/EMNIST, suffisant pour reconnaître des caractères individuels, et rapide à traiter.

6. **Pourquoi le nommage des fichiers comme labels ?** — Pas besoin de fichier d'annotations séparé. Le nom du fichier `AB123CD.png` contient directement le texte de la plaque.
