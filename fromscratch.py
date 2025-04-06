"""


"""
import math
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
	"""
	Fonction d'activation sigmoïde, tel que : 
	➝  f(x) = 1 / (1 + e^-x)
	"""
	return 1 / ( 1 + math.exp(-x) )


def seuil(x):
	"""
	On utilise : 
	seuil(x) = 1 si x > 0
	seuil(x) = 0 si x < 0
	seuil(0) = 0.5
	"""
	return 1 if x > 0 else 0 if x < 0 else 0.5


def prediction(x):
	"""
	➝ f(0) = 0.5, f(-∞) = 0, f(+∞) = 1
	"""
	return 1 if x > 0.5 else 0 if x < 0.5 else 0


class Perceptron():

	def __init__(self, nu = 0.1, epochs = 20):
		NB_INPUTS = 2
		self.nu = nu
		self.epochs = epochs
		self.w = np.random.uniform(-1, 1, size=(NB_INPUTS))
		self.bias = np.random.uniform(-1,1)
		self.frames_to_display = list()

	def activation(self, X):
		"""Fonction d'activation
		"""
		netz = X[0] * self.w[0] + X[1] * self.w[1] + self.bias
		return sigmoid(netz)

	def predict(self, X):
		oz = self.activation(X)
		return prediction(oz)

	def linear(self):
		#TODO
		# Équation : 0 = w1 * x1 + w2 * x2 + b
		# <=> x2 = -(w1/w2) * x1 - (b/w2)
		if self.w[1] == 0:
			raise ZeroDivisionError("w2 ne doit pas être nul pour tracer la droite")
		a = -self.w[0] / self.w[1]
		b = -self.bias / self.w[1]
		return a, b  # x2 = a * x1 + b

	def train(self, xi, tz, oz):
		dz = (tz - oz) * oz * (1 - oz)
		for i in range(len(self.w)):
			delta_w_iz = self.nu * dz * xi[i]
			self.w[i] += delta_w_iz

		delta_biais = self.nu * dz * 1
		self.bias += delta_biais

	def fit(self, X, z):
		# while (miss > 0):
		for _ in range(self.epochs):
		# miss = 2
		# while (miss > 0):
			miss = 0
			for xi, zi in zip(X,z):
				oz = self.activation(xi)
				self.train(xi, zi, oz)
				result = True if zi == prediction(oz) else False
				color = "green" if result else "red"
				miss += 0 if result else 1
				line = self.linear()
				self.frames_to_display.append([line, xi])
				



from sklearn import datasets
import animation

data = [
	[1,1,0],
	[2,0,0],
	[3,1,0],
	[0,1,0],
	[1,4,1],
	[0,2,1],
	[1,3,1],
	[0,1,1],
]

N = len(data)
data = np.array(data)
data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))

perceptron = Perceptron(epochs=100)
perceptron.fit(data, [0] * (N//2) + [1] * (N//2))

frames = perceptron.frames_to_display
animation.display(frames, data)
