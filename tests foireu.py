"""
Perceptron



notes :

initial ---
wxz=5
wyz=-3
wbz=1

Equation caractéristique :
5 x - 3 y + 1 = 0 
> 0 positif (+)
< 0 négatif (-)


Wxz = 5.0001
Wyz = -2.99
Wbz = 0.995 + 0.00001 = 0.995



"""

# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib.animation as animation

# def f(x, a=0.4):
#     return a * x + 2

# fig, ax = plt.subplots()
# x = np.arange(11)
# line, = ax.plot(x, f(x, 0))

# def update(frame):
#     a = frame * 0.2
#     line.set_ydata(f(x, a))
#     return line,

# ani = animation.FuncAnimation(fig, update, frames=9, interval=200)

# plt.show()


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation

def f(x, a=0.4):
    return a * x + 2

fig, ax = plt.subplots()
x = np.linspace(0, 10, 100)
line, = ax.plot(x, f(x))
ax.set_ylim(0, 5)  # Ajuste les limites de l'axe Y pour éviter les changements brusques

def update(frame):
    a = frame * 0.1
    line.set_ydata(f(x, a))
    return line,

ani = animation.FuncAnimation(
    fig, 
    update, 
    frames=range(20), 
    interval=200, 
    repeat=False
)

plt.show()


# ----------------------

from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron

X, y = load_digits(return_X_y=True)
clf = Perceptron(tol=1e-3, random_state=0)
clf.fit(X, y)
clf.score(X, y)
