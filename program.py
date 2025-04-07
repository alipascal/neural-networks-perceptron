import pandas
import numpy
import argparse

import perceptron
from perceptron import Perceptron
import animation


# Gestion des parametres de l'invite de commande
parser = argparse.ArgumentParser(description="Gestion des paramètres")
parser.add_argument('--file', help="Fichier CSV en entrée", default='example.csv')
args = parser.parse_args()

file = args.file

data = pandas.read_csv(file, sep=',')

# Début du progamme
N = len(data)
data = numpy.array(data)

data = perceptron.normalize(data)
p = Perceptron(epochs=100)
p.fit(data[:, :2],data[:, 2])

frames = perceptron.frames_to_display
animation.display(frames, data)