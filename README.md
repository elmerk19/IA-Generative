# IA Générative

Projet de TP consacré à l’intelligence artificielle générative appliquée à la reconnaissance et à la génération d’images manuscrites du dataset MNIST.

Ce dépôt contient un notebook de laboratoire (`lab2.ipynb`) dans lequel plusieurs modèles génératifs et d’analyse sont étudiés :

- un classifieur PyTorch pour la classification des chiffres manuscrits ;
- un autoencodeur pour apprendre une représentation latente ;
- un Variational Autoencoder (VAE) avec le reparametrization trick ;
- une visualisation de l’espace latent ;
- une génération d’images à partir d’un vecteur latent moyen.

## 1. Objectif du projet

L’objectif principal est d’explorer les principes de l’IA générative et de comprendre comment un modèle peut :

- encoder des images dans un espace latent compact ;
- reconstruire des images à partir de cette représentation ;
- régulariser cet espace latent via un VAE ;
- générer de nouvelles images plausibles à partir d’un espace latent appris.

## 2. Description du TP

Le projet s’appuie sur le dataset MNIST, qui contient des images en noir et blanc de chiffres écrits à la main (0 à 9). Les modèles sont entraînés sur ce dataset pour apprendre une représentation compacte des images puis générer des variations à partir de cette représentation.

Le notebook couvre successivement :

1. chargement des données MNIST ;
2. affichage d’exemples de chiffres ;
3. construction d’un classifieur simple ;
4. entraînement d’un classifieur pour atteindre une forte précision ;
5. mise en place d’un autoencodeur ;
6. apprentissage d’un espace latent ;
7. visualisation du nuage latent ;
8. génération d’images à partir d’un chiffre donné ;
9. construction et entraînement d’un VAE ;
10. calcul de la perte de reconstruction + KL divergence ;
11. reconstruction d’images avec le VAE ;
12. visualisation de l’espace latent appris.

## 3. Structure du dépôt

```bash
IA-Generative/
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
├── lab2.ipynb
├── generator_mnist.pt
├── data/
├── images_generees/
├── train-images-idx3-ubyte.gz
├── train-images-idx3-ubyte/
├── uv.lock
└── ...
```

### Détails

- `lab2.ipynb` : notebook principal contenant tout le TP et les expérimentations.
- `main.py` : fichier minimal d’entrée du projet.
- `pyproject.toml` : configuration Python du projet.
- `data/` : données MNIST téléchargées localement.
- `images_generees/` : images générées durant les expérimentations.
- `generator_mnist.pt` : éventuel checkpoint / modèle génératif entraîné.

## 4. Technologies utilisées

- Python 3.11+
- PyTorch
- Torchvision
- NumPy
- Matplotlib
- torcheval
- Jupyter / IPython

## 5. Prérequis

Assurez-vous d’avoir installé :

- Python 3.11 ou plus récent ;
- pip ou uv ;
- Git ;
- un environnement virtuel (recommandé).

## 6. Installation

### Option 1 : avec pip

```bash
git clone https://github.com/elmerk19/IA-Generative.git
cd IA-Generative
python -m venv .venv
source .venv/bin/activate   # Linux / macOS
# ou .venv\Scripts\activate  # Windows
pip install -e .
```

### Option 2 : avec uv

```bash
git clone https://github.com/elmerk19/IA-Generative.git
cd IA-Generative
uv sync
```

## 7. Lancement du projet

### Notebook

Ouvrez le notebook Jupyter :

```bash
jupyter notebook lab2.ipynb
```

ou dans VS Code, ouvrez le fichier `lab2.ipynb` et exécutez les cellules.

### Script Python

```bash
python main.py
```

Le script principal est très simple pour le moment et sert surtout de point d’entrée du projet.

## 8. Modèles présentés dans le TP

### 8.1 Classifieur

Un réseau simple basé sur une couche `Flatten`, une couche cachée `Linear(784, 128)`, puis une sortie `Linear(128, 10)`.

Objectif : classifier les chiffres manuscrits de 0 à 9.

Exemple de logique :

```python
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
```

### 8.2 Autoencodeur

Le modèle encode les images en un vecteur latent de faible dimension puis les reconstruit.

Architecture simplifiée :

- encodeur : 784 -> 128 -> 32 -> 2
- décodeur : 2 -> 32 -> 128 -> 784

Mise en œuvre : `nn.MSELoss()` et optimisation par Adam.

### 8.3 VAE (Variational Autoencoder)

Le VAE introduit une distribution latente gaussienne :

- moyenne `mu`
- variance logarithmique `logvar`

La fonction de perte est la somme de :

- perte de reconstruction : `binary_cross_entropy`
- perte KL : divergence de Kullback-Leibler

Formule conceptuelle :

```python
loss = rec_loss + beta * kl_loss
```

C’est cette régularisation qui permet de générer de nouvelles images plausibles dans l’espace latent.

## 9. Exemples de résultats attendus

Le notebook montre généralement :

- des images MNIST originales ;
- des images reconstruites par l’autoencodeur ;
- l’espace latent à 2 dimensions ;
- des images générées à partir d’un chiffre spécifique ;
- les performances d’un classifieur sur les données de test.

Les résultats peuvent inclure des scores de précision proches de 96 % sur le test set.

## 10. Explication du fonctionnement

### Prétraitement

Les images sont transformées en tenseurs `ToTensor()` pour être compatibles avec PyTorch.

### Entraînement

Les modèles utilisent :

- `DataLoader` pour charger les batchs ;
- `CrossEntropyLoss` pour la classification ;
- `MSELoss` pour l’autoencodeur ;
- `F.binary_cross_entropy` pour le VAE ;
- `Adam` comme optimiseur.

### Génération

Pour générer une image d’un chiffre donné :

- on encode toutes les images de ce chiffre ;
- on calcule le vecteur latent moyen ;
- on décode ce vecteur dans l’espace image.

## 11. Dataset utilisé

Le projet utilise le dataset MNIST, disponible par `torchvision.datasets.MNIST`.

Le téléchargement est automatique si la structure de données est absente :

```python
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
```

## 12. Commandes utiles

### Vérifier les dépendances

```bash
python -c "import torch, torchvision, numpy, matplotlib; print('OK')"
```

### Installer les dépendances

```bash
pip install torch torchvision matplotlib numpy torcheval
```

### Démarrer Jupyter

```bash
jupyter lab
```

## 13. Points d’amélioration

Le projet peut être amélioré avec :

- une meilleure architecture du VAE ;
- augmentation de la dimension latente ;
- ajout d’une interface utilisateur ;
- génération d’images plus réalistes ;
- sauvegarde des modèles en `.pt` ;
- comparaison entre plusieurs architectures ;
- ajout de tests unitaires.

## 14. License

Ce projet est fourni à titre pédagogique. La licence exacte dépend du contexte du cours ou des consignes de l’enseignant.

## 15. Conclusion

Ce TP illustre les fondements de l’IA générative avec des modèles simples mais instructifs :

- classification supervisée ;
- compression automatique via autoencodeurs ;
- régularisation par VAE ;
- génération d’images à partir d’un espace latent.

C’est un excellent exemple de projet pratique pour comprendre comment les modèles génératifs apprennent une représentation des données et produisent de nouvelles entrées réalistes.

## 16. Auteur / contexte

- Nom du projet : IA Générative
- Repository : `elmerk19/IA-Generative`
- Type : TP / laboratoire sur l’IA générative
- Domaine : deep learning, PyTorch, autoencodeurs, VAE, génération d’images

---

Vous pouvez maintenant utiliser ce fichier comme README complet du projet. Si vous voulez, je peux aussi vous proposer une version plus courte, plus académique, ou plus élégante pour un rendu de TP.
