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
- $w_{x_2z}$ : pareil avec $x_2$ et $z$
- $w_{bz}$ : le poids du bias

### Calculs apprentissage

Calculs résultat en sortie
$$
\begin{split}
somme = \Big( \displaystyle\sum^n w_j o_i \Big) + b_j \\
sortie = seuil(somme)
\end{split}

$$
➝ dans le cas d'une perceptron, on obtient $z$ en applicant la fonction $net_z$ (je crois)

Pour l'apprentissage le *perceptron* doit modifier la valeur de ces poids à chaque test effectuer. Pour déterminer ces modifications, on utilise les équations suivantes :

$$
\begin{equation}
net_z = w_{x_1z} \times x_1 - w_{x_2z} \times x_2 + w_{bz}
\end{equation}
$$
avec
$w_{ij}$ poids de la connexion $i j$ (weight)
$x_1, x_2$ entrées
$net_z$ valeurs intermédiaire de calcul

On spécifiie la formule du cas général $(1)$ à l'équation $(2)$


$t_z$ est la valeur attendue (target)
$o_z$ sortie (output)
$d_k$ « signal d’erreur » pour un neurone $k$ en sortie du réseau
$d_z$ signal d'erreur du seul neurone en sortie $z$
$\Delta w_{ij}$ signifie que l’on modifie $w_{jk}$ d’une valeur égale à celle située à droite du signe = de l’équation
$\nu$ est le pas d’apprentissage

$$
\Delta w_{ij} = \nu \times d_k \times o_j
$$
$$
\begin{split}
& d_k = (t_k – o_k ) f’(net_k) \\
& d_j = f’(net_j) \sum_{k} d_k w_{jk} \\
& d_z = (t_z - o_z) \times f'(net_z) \\
& d_z = (t_z - o_z) \times o_z(1 - o_z)

\end{split}
$$
$$
\begin{split}
	&f'(net_z) = f(net_z)(1 - f(net_z)) \\
    &f'(net_z) = o_z(1 - o_z)
\end{split}
$$

avec $o_z = f(net_z)$

$$
d_z = (t_z - O_z) \times O_z(1 - O_z)
$$



$$
\begin{split}
& Perceptron = 1 neuron \\
& Inputs X \\
& Weights W \\
& Output O \\
& O = f(X . W) \\
& f = sigmoid \\
\end{split}
$$
Calculs erreurs
$$
\Delta w_{jk} = \nu \times d_k \times o_j
$$
