import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Liste d'instructions : chaque élément est un tuple ("type", données)
# "line": (a, b), "point": (x, y)
instructions = [
    ("line", (-1, 0)),
    ("point", (2, 1)),
    ("line", (0.5, 1)),
    ("point", (-3, -2)),
    ("line", (1, -1)),
    ("point", (0, 0))
]

# Création de la figure et des axes
fig, ax = plt.subplots()
# Abscisses utilisées pour tracer les droites
x = np.linspace(-10, 10, 200)

# Liste pour stocker les objets Line2D (droites) et les points
lines = []
points = []

# Fonction d'initialisation de l'animation (appelée une seule fois au début)
def init():
    ax.set_xlim(-10, 10)   # Limites de l'axe x
    ax.set_ylim(-10, 10)   # Limites de l'axe y
    return []              # Rien à afficher encore

def update(frame):
    item_type, data = instructions[frame]
    if item_type == "line":
        a, b = data
        y = a * x + b
        line, = ax.plot(x, y, label=f'{a:.2f}x + {b:.2f}', color='blue')
        lines.append(line)  # Garde une référence à chaque ligne
        return lines + points  # Retourne toutes les droites et points actuels
    elif item_type == "point":
        xp, yp = data
        point = ax.scatter(xp, yp, color='red', s=50)
        (points.pop() for elem in points)
        points.append(point) # Garde une référence pour les points
        print(points)
        return lines + points  # Retourne toutes les droites et points actuels

# Création de l'animation : une frame par droite, sans boucle
ani = animation.FuncAnimation(
    fig,               # La figure à animer
    update,            # La fonction de mise à jour
    frames=len(instructions),  # Nombre de frames (une par droite)
    init_func=init,    # Fonction d'initiation
    blit=False,        # Enlève blit pour garder tous les objets affichés
    repeat=False       # Ne pas boucler l'animation
)

plt.legend()
plt.show()

"""
image 1 : ("line", (-1, 0))
image 2 : ("line", (-1, 0)), ("point", (2, 1))
image 3 : ("line", (0.5, 1)), ("point", (2, 1))
image 4 : ("line", (0.5, 1)), ("point", (-3, -2))
image 5 : ("line", (1, -1)), ("point", (-3, -2))
...
...
    ,
    ("point", (0, 0))
"""