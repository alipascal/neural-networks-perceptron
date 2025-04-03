##### Alicia TCHEMO
Apprentissage Machine - Master 1 INFO; Données, Connaissance, Intelligence (DCI); Université Paris Cité
alicia.tchemo@etu.u-paris.fr
*Réalisé en mars 2025*

==TODO lien GitHub==

---

## Abstract

blabla


## Sommaire
1. Description de la méthode
2. Implémentation & Documentation du Programme
3. Analyse des résultats
4. Conclusion


Calculs apprentissage
$$
net_z = w_{x_1z} \times x_1 - w_{x_2z} \times x_2 + w_{bz}
$$

$w_{ij}$ poids de la connexion $i j$ (weight)
$t_z$ est la valeur attendue (target)
$o_z$ sortie (output)
$d_k$ « signal d’erreur » pour un neurone $k$ en sortie du réseau
$d_z$ signal d'erreur du seul neurone en sortie $z$
$\Delta w_{ij}$ signifie que l’on modifie $w_{jk}$ d’une valeur égale à celle située à droite du signe = de l’équation
$\nu$ est le « pas » d’apprentissage

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

Calculs résultat en sortie
$$
\begin{split}
somme = \Big( \displaystyle\sum^n w_j o_i \Big) + b_j \\
sortie = z = seuil(somme)
\end{split}

$$
➝ dans le cas d'une perceptron, on obtient $z$ en applicant la fonction $net_z$ (je crois)

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
\Delta w_{jk} = v \times d_k \times o_j
$$
