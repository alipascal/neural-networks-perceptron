"""


class Perceptron():

	def predict(self, X):
        w = self.w
        b = self.bias
        z = sigmoid(np.dot(X,self.w) + b)

    def display_netz(self):
    	pass



"""
import math
import numpy as np



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

	def __init__(self,nb_inputs=2, lr = 0.01, epochs = 20):
		#setting default parameters
		#TODO
		self.lr = lr
		self.epochs = epochs
		self.nb_inputs = nb_inputs
		self.w = np.random.uniform(-1, 1, size=(nb_inputs))
		self.bias = random.uniform(-1,1)
		self.misses = []

	def predict(self, X):
		w = self.w
		b = self.bias
		netz = np.dot(X,self.w) + b
		oz = sigmoid(netz)
		tz = seuil(oz)
		return tz

	def display_netz(self):
		pass

	def learn(self, xi, zi):
		z = self.predict(xi)
		if zi != z:
			"le test est mal placé"
		delta_w_iz = nu * dz * oz
		dz = (tz - oz) * oz * (1 - oz)

		#TODO

	def fit(self, X, z):
		# pour tous xi, yi de zip(X,y)

		# self.learn(xi, yi)
		pass


# ----------------------------
 		
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

# -------------------


# import numpy as np
import matplotlib.pyplot as plt

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
plt.show()
