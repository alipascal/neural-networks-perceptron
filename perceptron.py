"""
@author Alicia TCHEMO
@date 2025-04-08
Apprentissage Machine - M1 INFO DCI - Université Paris-Cité
"""

import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x:int) -> float:
	"""
	Fonction d'activation sigmoïde, tel que : 
	➝  f(x) = 1 / (1 + e^-x)
	"""
	return 1 / ( 1 + math.exp(-x) )


def seuil(x:int) -> float:
	"""(pas utilisé)
	Fonction d'activation d'approxiamtion : TODO
	➝	seuil(x) = 1 si x > 0
		seuil(x) = 0 si x < 0
		seuil(0) = 0.5
	"""
	return 1 if x > 0 else 0 if x < 0 else 0.5


def prediction(x:float) -> int:
	"""Retourne la classe prédite, tel que :
	➝ f(0) = 0.5, f(-∞) = 0, f(+∞) = 1
	"""
	return 1 if x > 0.5 else 0 if x < 0.5 else 0

def normalize(data):
	"""Normalisation des données entre 0 et 1"""
	return (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))


class Perceptron():

	def __init__(self, nu = 0.1, epochs = 1000):
		NB_INPUTS = 2
		self.nu = nu
		self.epochs = epochs
		self.w = np.random.uniform(-1, 1, size=(NB_INPUTS))
		self.bias = np.random.uniform(-1,1)
		self.frames_to_display = list()

	def activation(self, X):
		"""Fonction d'activation"""
		netz = X[0] * self.w[0] + X[1] * self.w[1] + self.bias
		return sigmoid(netz)

	def predict(self, X):
		oz = self.activation(X)
		return prediction(oz)

	def linear(self):
		"""Calcule les coefficients a et b de la droite de décision, tel que :
		0 = w1 * x1 + w2 * x2 + bias
		<=> x2 = -(w1/w2) * x1 - (bias/w2)
		donc on a, x2 = a * x1 + b
		"""
		if self.w[1] == 0:
			raise ZeroDivisionError("Impossible d'afficher la droite. w_x2z ne doit par être égal à 0.")
		a = -self.w[0] / self.w[1]
		b = -self.bias / self.w[1]
		return a, b  

	def train(self, xi, tz, oz):
		dz = (tz - oz) * oz * (1 - oz)
		# Modifier le poids entre les entrées et le neurone z
		for i in range(len(self.w)):
			delta_w_iz = self.nu * dz * xi[i]
			self.w[i] += delta_w_iz
		# Modifier le poids du bias
		delta_biais = self.nu * dz * 1
		self.bias += delta_biais

	def fit(self, X, z):
		for n in range(self.epochs + 1):
			miss = 0
			for xi, zi in zip(X,z):
				oz = self.activation(xi)
				self.train(xi, zi, oz)
				result = True if zi == prediction(oz) else False
				color = "green" if result else "red"
				miss += 0 if result else 1
				line = self.linear()
				self.frames_to_display.append([line, xi, n])
			print(f"\repochs={n}", end="")
			if miss == 0:
				break
		print()

				

# --- test ------------------------------------------------------------------------

import animation

if __name__ == '__main__':
	data = [
		[2,1,0],
		[2,0,0],
		[3,1,0],
		[1,1,0],
		[1,4,1],
		[0,2,1],
		[1,3,1],
		[0,1,1],
	]

	N = len(data)
	data = np.array(data)

	data = normalize(data)
	perceptron = Perceptron(epochs=500)
	perceptron.fit(data[:, :2],data[:, 2])

	# Tests animation
	frames = perceptron.frames_to_display
	animation.display(frames, data, millisecondes=50)
