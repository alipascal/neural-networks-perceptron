# Visualisation de l'apprentissage d'un Perceptron

Implémentation d'un perceptron à deux paramètres permettant d'observer graphiquement l'évolution de son apprentissage. Ce projet visualise la modification des poids et de la frontière de décision durant l'entraînement.

<p align="center">
  <img width="80%" alt="ANIM1 figure nu=0.01 max=10000" src="./results/ANIM1_figure_0.01_10000.gif" />
</p>


## Fonctionnalités

* Lecture de fichiers CSV au format (id, x1, x2, classe)
* Animation de la progression de l'entraînement
* Affichage du nombre d'epochs et du temps d'exécution
* Export de l'animation au format GIF


## Fonctionnement

### Architecture

<p align="center">
  <img width="80%" alt="perceptron-2-inputs-NB" src="https://github.com/user-attachments/assets/e48e28f6-666c-4631-8ee2-b720a2968ad7" />
</p>

Le perceptron implémente un réseau à deux entrées ($x_1$, $x_2$) avec les équations suivantes :

$$net_z = w_{x_1z} \times x_1 + w_{x_2z} \times x_2 + w_{bz}$$

$$sigmoid = f(x) = \frac{1}{1 + e^{-x}}$$

$$\Delta w_{iz} = \nu \times d_z \times x_i$$

### Méthodes principales

**`Perceptron().fit(X,y)`** : Entraîne le modèle sur les données
- Appel par défaut : `Perceptron(nu=0.1, epochs=10000).fit(X,y)`
- `nu` : taux d'apprentissage
- `epochs` : nombre maximal d'itérations

**`display(frames, data)`** : Génère l'animation de l'apprentissage
- Appel par défaut : `display(frames, data, millisecondes=50, save=False)`
- `frames` : étapes successives (positions du curseur et droite de séparation)
- `millisecondes` : intervalle entre chaque frame
- `save` : sauvegarde en GIF si True

### Algorithme

```
Normalisation des données d'entrée
Initialisation aléatoire des poids

Tant que epochs_max non atteint :
    Pour chaque point :
        Application de la fonction d'activation
        Mise à jour des poids
        Vérification de la prédiction
    
    Si aucune erreur : arrêt de l'entraînement

Affichage de l'animation
```

### Résultats

Statistiques sur 100 exécutions avec `data.csv` :

| Taux d'apprentissage | Epochs moyen | Epochs médian | Epochs max | Epochs min | Temps moyen |
|:--------------------:|:------------:|:-------------:|:----------:|:----------:|:-----------:|
| ν = 0.1              | 1008.62      | 1019.0        | 1040       | 0          | 0.2 s       |
| ν = 0.01             | 8417.2       | 8842.0        | 9133       | 58         | 1.4 s       |

## Installation

### Prérequis

- Python
- Packages :
  - `pandas` : pour la lecture du fichier CSV
  - `numpy`
  - `matplotlib` : pour l'affichage de l'animation

```shell
pip install -r requirements.txt
```

### Utilisation

```shell
python program.py                    # Utilise data.csv par défaut
python program.py <mon_fichier.csv>  # Fichier personnalisé
```

**Format CSV requis :**
```csv
id,x1,x2,classe
1,...,...,...
2,...,...,...
...
```
