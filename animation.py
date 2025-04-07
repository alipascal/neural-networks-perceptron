"""
@author Alicia TCHEMO
@date 2025-04-08
Apprentissage Machine - M1 INFO DCI - Université Paris-Cité
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def display(frames, data):
    """
    Affiche une animation de l'apprentissage d'un perceptron

    :param data (list): Les données classées en + et - a affiché sur le graphe 
    :param frames (list(list(tuple))): Liste des états à afficher, contenant les valeurs des éléments en animation
    """

    class UpdateAnimation:

        def __init__(self, ax, data):
            self.success = 0
            self.line, = ax.plot([], [], 'b-')
            self.point = ax.scatter(0, 0, color="red", s=50, alpha=0.5)
            self.nb_tests = 0
            self.text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
            self.x = np.linspace(-1, 2, 200)
            self.ax = ax

            # affichage des axes 
            self.ax.set_xlim(-1, 2)
            self.ax.set_ylim(-1, 2)

            # affichage des points
            N = len(data)
            s = data[:, :2] # set
            c = data[:, 2] # classification
            self.ax.scatter(s[c == 0, 0], s[c == 0, 1], marker='+', color='black')
            self.ax.scatter(s[c == 1, 0], s[c == 1, 1], marker='_', color='black')

            # titres & labels
            self.ax.set_title("Perceptron learning")
            self.ax.set_xlabel("x1")
            self.ax.set_ylabel("x2")

        def start(self):
            return self.line, self.point, self.text

        def __call__(self, frame):
            line, point = frame[0], frame[1]
            
            # update de la droite
            a, b = line
            y = a * self.x + b
            self.line.set_data(self.x, y)

            # upadte du curseur
            posx, posy = point[0], point[1]
            self.point.set_offsets(np.array([[posx, posy]]))

            # update nb_tests text
            self.nb_tests += 1
            self.text.set_text(f"nb tests = {self.nb_tests}")

            return self.line, self.point, self.text

    fig, ax = plt.subplots()
    update = UpdateAnimation(ax, data)
    anim = FuncAnimation(fig, update, init_func=update.start, frames=frames, interval=100, blit=True, repeat=False)
    plt.show()



# --- test ------------------------------------------------------------------------

if __name__ == '__main__':

    frames = [
        [(-1, 0), (3, 1)],
        [(0.5, 1), (2, 0.5)],
        [(1, -1), (1, 1)],
        [(0.5, 1), (6, 2)],
    ]
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
    display(frames, data)