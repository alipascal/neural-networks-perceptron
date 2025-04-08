"""
@author Alicia TCHEMO
@date 2025-04-08
Apprentissage Machine - M1 INFO DCI - Université Paris-Cité

Programme d'un perceptron à deux paramètres d'entrée, avec visualisation de l'évolution de son apprentissage via une animation.

Fichier Main !!!
"""

import pandas
import numpy
import argparse
from time import time

import perceptron
from perceptron import Perceptron
import animation


# Gestion des parametres de l'invite de commande
parser = argparse.ArgumentParser(description="Gestion des paramètres")
parser.add_argument('file', help="Fichier CSV en entrée", nargs='?', default='data.csv')
args = parser.parse_args()

file = args.file

# Lecture du fichier
data = pandas.read_csv(file, sep=',')
id_col = data.columns[0]
data.set_index(id_col, inplace=True)
    # Numérisation (string to bool)
classification = data.columns[-1]
d = None
if data[classification].dtype != 'int64':
    d = {elem: index for index, elem in enumerate(pandas.unique(data[classification]))}
    data[classification] = data[classification].map(d)

# Début du progamme
N = data.shape[0]
data.iloc[:, :2] = perceptron.normalize(data.iloc[:, :2])

do_test = None
if "?" in data.iloc[-1]:
    do_test = True

if do_test:
    test = data.iloc[-1]
    data = data.iloc[:-1]

p = Perceptron(epochs=1000, nu=0.01)
data = numpy.array(data)
print("Début de l'entrainement du perceptron")
t = time()
p.fit(data[:, :2],data[:, 2])
print(f"Fin entrainement ({time() - t}s)")

if do_test:
    test = numpy.array(test)
    result = p.predict(test[0:2])
    print(result if d == None else next(k for k, v in d.items() if v == result))

frames = p.frames_to_display
animation.display(frames, data, millisecondes=1)