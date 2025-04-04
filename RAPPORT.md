##### Alicia TCHEMO
Apprentissage Machine - Master 1 INFO; Données, Connaissance, Intelligence (DCI); Université Paris Cité
alicia.tchemo@etu.u-paris.fr
*Réalisé en mars 2025*

==TODO lien GitHub==

---

## Abstract

Ce rapport contient le compte rendu d'un programme Python [^1] qui, par un noeud de réseaux de neurone unique appelé *perceptron*, permet la classification des pays d'Afrique et d'Europe. De plus, l'entrainement de perceptron est visualisable à travers un graphique animé. 

## Sommaire
1. Description de la méthode
2. Implémentation & Documentation du Programme
3. Analyse des résultats
4. Conclusion

## 1. Description

Pour l'implémentation, d'abord, on se réfère aux équations de modélisation de réseau de neurone, pour ensuite les simplifié. 

***Figure 1.*** Réseaux de neurone

Dans ce cas, on considère uniquement deux variables d'entrées $x_1$, $x_2$, tel que $X = \{ x_1, x_2 \}$, et un neurone de sortie $z$.


***Figure 2.*** Perceptron à deux entrées

Le poids des connexions entre les noeuds généralement noté $w_{ij}$, ici sont réduite au nombre de trois, avec : 
- $w_{x_1z}$ : connexion entre la variable d'entrée $x_1$ et le neurone de sortie $z$
- $w_{x_2z}$ : même chose avec $x_2$ et $z$
- $w_{bz}$ : le poids du biais

#### Calculs d'apprentissage

==TODO== La version met à jour W après avoir vu chaque exemple.
Pour l'apprentissage le *perceptron* on identifie un seul neurone de sortie, et aucun couche cachée de neurone. Et si on applique l'apprentissage « *un par un* », les valeur de chaques poids s'effectue à chaque exemple test. Pour évalué ces modifications, on utilise les équations ci-dessous.

On spécifie la formule du cas général pour obtenir l'équation $(1)$.
$$
\begin{split}
somme = \Big( \displaystyle\sum^n w_j o_i \Big) + b_j \\
\end{split}
$$
-> voir cours
$(1)$
$$
\begin{equation}
net_z = w_{x_1z} \times x_1 - w_{x_2z} \times x_2 + w_{bz}
\end{equation}
$$
Avec,
- $w_{ij}$ poids de la connexion $i$ et $j$ 
- $x_1, x_2$ entrées
- $net_z$ valeurs intermédiaire de calcul



Avec la fonction sigmoïde,
$(2)$
$$
sigmoid = f(x) = 
$$

on a les résultats suivants à implémenter,

$$
\begin{split}
& d_k = (t_k – o_k ) f’(net_k) \\
& d_j = f’(net_j) \sum_{k} d_k w_{jk} \\
\end{split}
$$
donc, avec,
$$
\begin{split}
output &= f(inputs.weights) \\
o_z &= f(net_z)
\end{split}
$$

et,
$$
\begin{split}
	&f'(net_z) = f(net_z)(1 - f(net_z)) \\
    &f'(net_z) = o_z(1 - o_z)
\end{split}
$$
on a,
$(3)$
$$
d_z = (t_z - o_z) \times o_z(1 - o_z)
$$

Avec,
- $t_z$ est la valeur attendue (*target*)
- $o_z$ valeur de sortie (*output*)
- $d_z$ signal d'erreur du neurone $z$ en sortie du réseau

D'après ces équations et ce résonnement (3), ont applique la formule pour déterminer les valeurs des poids à modifier pour l'apprentissage du perceptron :

$$
\Delta w_{ij} = \nu \times d_k \times o_j
$$
soit,
$(4)$
$$
\Delta w_{iz} = \nu \times d_z \times o_z
$$

L'apprentissage "*un par un*" met à jour les poids $w_{x_iz}$ après avoir vu chaque exemple test. L'apprentissage se termine lorsqu'il n'y a plus d'erreur, soit de différence entre la valeur attendu $t_z$ et la valeur de sortie $o_z$.


#### Données d'apprentissage

on souhaite classer des pays pour savoir si 
ou couleur chaude & couleur froides

## 2. Implémentation

## 3. Analyse des résultats

## 4. Conclusion
