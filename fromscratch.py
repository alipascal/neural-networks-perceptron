"""

Perceptron() fdrom scratch
https://python.plainenglish.io/building-a-perceptron-from-scratch-a-step-by-step-guide-with-python-6b8722807b2e

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
	On utilise pas : 
	seuil(x) = 1 si x > 0
	seuil(x) = 0 si x < 0
	seuil(0) = 0.5
	➝ 1 if x > 0 else 0 if x < 0 else 0.5
	"""
	return 1 if x > 0 else 0 if x < 0 else 0.5


def prediction(x):
	"""
	➝  f(0) = 0.5, f(-∞) = 0, f(+∞) = 1
	"""
	return 1 if x > 0.5 else 0 if x < 0.5 else 0


class Perceptron():

	def __init__(self,nb_inputs=2, nu = 0.01, epochs = 20):
		self.nu = nu
		self.epochs = epochs
		self.nb_inputs = nb_inputs
		self.w = np.random.uniform(-1, 1, size=(nb_inputs))
		self.bias = np.random.uniform(-1,1)
		self.misses = []

	def activation(self, X):
		"""Fonction d'activation
		"""
		netz = np.dot(X,self.w) + self.bias
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
		return (a, b)  # x2 = a * x1 + b

	def train(self, xi, tz, oz):
		dz = (tz - oz) * oz * (1 - oz)
		for i in range(len(self.w)):
			delta_w_iz = self.nu * dz * xi
			self.w[i] += delta_w_iz
		delta_biais = self.nu * dz
		self.bias += delta_biais

	def fit(self, X, z):
		oz = self.activation(X)
		for xi, zi in zip(X,z):
			self.train(xi, zi, oz)
			result = True if zi == prediction(oz) else False
			color = "green" if result else "red"

# ----------------------------
def test_display():
	X = (0, 1)
	z = 1

	wx1z = 5
	wx2z = -3
	wbz = 1

	w = [wx1z, wx2z]


	netz = np.dot(X,w) + wbz
	oz = sigmoid(netz)
	tz = z
	pred_z = seuil(netz)

	color_valid = "grey"
	if pred_z == tz:
		color_valid = "green"
		print(f"le point {X} est bien placé")
	else:
		color_valid = "red"
		print(f"le point {X} est mal placé")

	dz = (tz - oz) * oz * (1 - oz)

	print(f"point{X}, netz={netz}, oz={oz}, pred_z={pred_z}, tz={tz}, dz={dz}")

	# x = list(range(2))
	# y = [np.dot((xi, 1), w) + wbz for xi in x]
	x = np.linspace(-2, 2, 400)
	y = (wx1z * x + wbz) / -wx2z

	fig, ax = plt.subplots()
	ax.set_aspect('equal', adjustable='box')

	ax.plot(x, y)


	# Selector
	plt.scatter(X[0], X[1], alpha=0.5, color=color_valid, s=200)

	# Points 
	plt.scatter(0, 1, color="black", marker='+')
	plt.scatter(0, 0, color="black", marker="_")
	plt.scatter(1, 0, color="black", marker="+")
	plt.scatter(1, 1, color="black", marker="+")
	plt.show()

# Arrow
# xp, yp = 5, np.dot((5, 1), w) + wbz # Point pour la flèche
# dx, dy = 1, (np.dot((6, 1), w) + wbz) - yp # Vecteur directeur de la droite
# # Vecteur normal (perpendiculaire)
# normal_x, normal_y = -dy, dx
# normal_x, normal_y = normal_x / np.hypot(normal_x, normal_y), normal_y / np.hypot(normal_x, normal_y)
# # Dessin de la flèche perpendiculaire
# ax.arrow(xp, yp, normal_x, normal_y, head_width=1, head_length=1, fc='r', ec='r', label="Flèche perpendiculaire")
# ax.legend()



# ax.set_xlim(min(x), max(x))
# ax.set_ylim(min(y), max(y))

