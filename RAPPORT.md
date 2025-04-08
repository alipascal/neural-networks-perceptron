##### Alicia TCHEMO
Apprentissage Machine - Master 1 INFO; Données, Connaissance, Intelligence (DCI); Université Paris Cité
alicia.tchemo@etu.u-paris.fr
*Réalisé en mars 2025*

[Programme Python accessible sur GitHub](https://github.com/alipascal/neural-networks-perceptron)

---
## Abstract

Ce rapport contient le compte rendu d'un programme Python [^1] qui, par un noeud de réseaux de neurone unique appelé *perceptron*, permet la classification des pays d'Afrique et d'Europe. De plus, l'entrainement de perceptron est visualisable à travers un graphique animé. 

## Sommaire
1. Description de la méthode
2. Implémentation & Documentation du Programme
3. Analyse des résultats
4. Conclusion

## 1. Description

#### Problématique

Malheureusement, Pierre s'est réveillé dans une ville dont il ne connais pas le nom. Il souhaite savoir si il se situe en Belgique ou en France. Or, il ne connait uniquement sa position géographique (longitude et latitude), et celle de 25 autres villes.
Pour aidez Pierre, on réalise un réseau de neurone à un seul neurone (*perceptron*), pour classifié les villes en deux pays, et prédire la position de Pierre selon le model.

#### Théorie

Pour l'implémentation, d'abord, on se réfère aux équations de modélisation de réseau de neurone, pour ensuite les simplifié. 

![[neural network single output.png]]

***Figure 1.*** Réseaux de neurone

Dans ce cas, on considère uniquement deux variables d'entrées $x_1$, $x_2$, tel que $X = \{ x_1, x_2 \}$, et un neurone de sortie $z$.

![[a multi-unit perceptron.png]]

***Figure 2.*** Perceptron à plusieurs entrées [^3]

![[perceptron.png]]

***Figure 3.*** Perceptron à deux entrées

Le poids des connexions entre les noeuds généralement noté $w_{ij}$, ici sont réduite au nombre de trois, avec : 
- $w_{x_1z}$ : connexion entre la variable d'entrée $x_1$ et le neurone de sortie $z$
- $w_{x_2z}$ : même chose avec $x_2$ et $z$
- $w_{bz}$ : le poids du biais

#### Calculs d'apprentissage

==TODO== La version met à jour W après avoir vu chaque exemple.
Pour l'apprentissage le *perceptron* on identifie un seul neurone de sortie, et aucun couche cachée de neurone. Et si on applique l'apprentissage « *un par un* », les modifications sur de chaques poids s'effectue à chaque exemple testé. Pour évalué ces modifications, on utilise les équations ci-dessous.

On spécifie la formule du cas général pour obtenir l'équation $(1)$.
$$
\begin{split}
somme = \Big( \displaystyle\sum^n w_j o_i \Big) + b_j \\
\end{split}
$$
[^2]

$$
\begin{equation}
net_z = w_{x_1z} \times x_1 - w_{x_2z} \times x_2 + w_{bz}
\end{equation}
$$
***Equation*** $(1)$.

Avec,
- $w_{ij}$ poids de la connexion $i$ et $j$ 
- $x_1, x_2$ entrées
- $net_z$ valeurs intermédiaire de calcul


Avec la fonction sigmoïde,
$$
sigmoid = f(x) = \frac{1}{1 + e^{-x}}
$$
***Equation*** $(2)$.

D'après les équation suivante :

$$
\begin{split}
& d_k = (t_k – o_k ) f’(net_k) \\
& d_j = f’(net_j) \sum_{k} d_k w_{jk} \\
\end{split}
$$
$$
\begin{split}
output &= f(inputs.weights) \\
o_z &= f(net_z)
\end{split}
$$
***Equation*** $(3)$.
$$
\begin{split}
	&f'(net_z) = f(net_z)(1 - f(net_z)) \\
    &f'(net_z) = o_z(1 - o_z)
\end{split}
$$
On implémente :
$$
\begin{equation}
d_z = (t_z - o_z) \times o_z(1 - o_z)
\end{equation}
$$
***Equation*** $(4)$.

En sachant que,
- $t_z$ est la valeur attendue (*target*)
- $o_z$ valeur de sortie (*output*)
- $d_z$ signal d'erreur du neurone $z$ en sortie du réseau

D'après ces équations et ce résonnement, ont applique la formule pour déterminer les valeurs des poids à modifier pour l'apprentissage du perceptron :

$$
\Delta w_{ij} = \nu \times d_k \times o_j
$$
soit,

$$
\Delta w_{iz} = \nu \times d_z \times x_i
$$
***Equation*** $(5)$.

L'apprentissage "*un par un*" met à jour les poids $w_{x_iz}$ après avoir vu chaque exemple test. L'apprentissage se termine lorsqu'il n'y a plus d'erreur.

#### Données d'apprentissage

on souhaite classer des pays pour savoir si 
ou couleur chaude & couleur froides
soit de différence entre la valeur attendu $t_z$ et la valeur de sortie $o_z$.

## 2. Implémentation

#### Exécution

Voici la ligne de commande pour tester le programme :

```shell
python program.py
python program.py test.csv
python program.py nom_du_fichier.csv
```

#### Dépendances

```shell
pip install -r requirements.txt
```

- `pandas`
- `numpy`
- `matplotlib`

#### Méthodes

On peut expliquer la structure du programme en plusieurs étapes :

- Conversion du fichier entré en donnés exploitable
- Initialisation d'un perceptron avec des valeurs de poids aléatoire
- Entrainement "*un par un*" du perceptron selon le nombres d'itération *epochs* (avec sauvegarde des étapes d'apprentissage)
- Affichage de l'animation selon les étapes d'apprentissage données

Il y a d'implémenté les 5 équations.
*➝ voir 1. Description - Calculs d'apprentissage*

Le programme s'arrête soit lorsque le nombre d'itération maximal (nombre *epochs*) est atteint, soit lorsque le modèle ne commet plus aucune erreur de prédictions.

On identifie deux méthodes fondatrice dans le programme : 
- `Perceptron().fit(X,y)` : qui permet au model de s'entrainer sur le jeu de données entrées
- `display(frames, data)` : qui permet le visualisation de l'animation

Par défaut, ces méthodes sont fixé, tel que :

`Perceptron(nu = 0.1, epochs = 20).fit(X,y)`, avec `nu` comme taux d'apprentissage, et `epochs` le nombre d'itérations

`display(frames, data, millisecondes=50, save=False)`, avec `frames` comme liste des étapes de l'apprentissage (positions successives du curseur rouge, et de la droite de décision), et `data` contient les données utilisées pour l'affichage des points, et avec `millisecondes` comme durée entre chaque frame de l'animation, et `save` un booléen pour la sauvegarde de l'animation au format .gif

#### Fichiers d'entrée

Pour résoudre la problématique, on utilise les données : 
```csv
ville,longitude,latitude,pays
Paris,2.349014,48.864716,France
Amiens,2.295753,49.894066,France
...
...
Bruxelles,4.351721,50.850346,Belgique
Liège,5.5685,50.6326,Belgique
...
...
```

Cependant, le programme est compatible avec tous les fichiers d'entrée qui respectent le format ci-dessous : 
```csv
id,x1,x2,classe
1,2,1,0
2,2,0,0
...
...
```
*➝ voir 2. Implémentation - Exécution*
## 3. Analyse des résultats

#### Prédiction

On test le modèle avec les coordonnes de Pierre qui sont les suivantes :
	longitude = 0.072
	latitude = 49.356

D'après le programme, Pierre se situe en **France**.
```
Début de l'entrainement du perceptron
Fin entrainement (0.16190838813781738s)
France
```

#### Visualisation de l'apprentissage

Lorsque de l'exécution du programme avec les données du fichier par défaut `data.csv` (contenant la position latitude et longitude de 25 villes de Belgique et de France), on obtient le résultat si dessous :

En sachant que  $\nu = 0.01$ et $epochs = 1000$ (soit le max d'itérations possibles).



Avec les données tests du fichier `test.csv`, et avec $\nu = 0.1$, on obtient un apprentissage comme ci-dessous :  
![[figureT1.gif]]
#### Erreurs

Sur le test ci-dessous, qui représente la dernière itération de l'entrainement du perceptron sur des données test (`test.csv`), on remarque que la droite de décision est mal placé.
L'erreur est sûrmeent du à la modification systématique des valeurs des poids (*un par un*) même si les tests sont bien placé.
Pour résoudre ce problème il faudrait modifié les poids du perceptron à la fin d'une itération *epochs*, c'est-à-dire lorsque tout les points on été vu au moins une fois.
![[Figure0.png]]

## 4. Conclusion

#### Résolution

Finalement, Pierre à observer l'environement autour de lui, et à compris qu'il se situais à Deauville, en France. Plus de peur que de mal pour lui, qui n'a pas eu besoin d'utilisé le model de perceptron. Après une petite soirée mouvementée, on ne sait jamais trop se situer.


## Références

[^1]: Programme Python "Visualisation de l'apprentissage d'un Perceptron", sur [GitHub](https://github.com/alipascal/neural-networks-perceptron) 

[^2]: Bruno Bouzy, Cours Apprentissage Machine "Induction on Decision Trees", diapo 13–15, (2025)

[^3]: Marius-Constantin POPESCU, Valentina E. BALAS, Liliana PERESCU-POPESCU3, Nikos MASTORAKIS; "Multilayer Perceptron and Neural Networks"; Issue 7, Volume 8, July (2009)