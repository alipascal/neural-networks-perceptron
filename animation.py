"""
@author Alicia TCHEMO
@date 2025-04-08
Apprentissage Machine - M1 INFO DCI - Université Paris-Cité
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def display(frames, data, millisecondes=50, save=False):
    """
    Affiche une animation de l'apprentissage d'un perceptron

    :param data (list): Les données classées en + et - a affiché sur le graphe 
    :param frames (list(list(tuple))): Liste des états à afficher, contenant les valeurs des éléments en animation
    :param-optionnel millisecondes (int): Temps d'intervale entre chaque frame de l'aniamation
    :param-optionnel save (bool): Pour la sauvegarde en format .gif de l'animation
    """

    class UpdateAnimation:

        def __init__(self, ax, data):
            self.success = 0
            self.line, = ax.plot([], [], 'b-')
            self.point = ax.scatter(0, 0, color="red", s=90, alpha=0.4)
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
            self.ax.set_xlabel(r"$x_1$")
            self.ax.set_ylabel(r"$x_2$")

        def start(self):
            return self.line, self.point, self.text

        def __call__(self, frame):
            line, point, epochs = frame[0], frame[1], frame[2]
            
            # update de la droite
            a, b = line
            y = a * self.x + b
            self.line.set_data(self.x, y)

            # upadte du curseur
            posx, posy = point[0], point[1]
            self.point.set_offsets(np.array([[posx, posy]]))

            # update nb_tests text
            self.text.set_text(f"epochs = {epochs}")

            return self.line, self.point, self.text

    fig, ax = plt.subplots()
    update = UpdateAnimation(ax, data)
    anim = FuncAnimation(fig, update, init_func=update.start, frames=frames, interval=millisecondes, blit=True, repeat=False)
    if save:
        writer = PillowWriter(fps=15,metadata=dict(artist='me'),bitrate=1800)
        anim.save('figure.gif', writer=writer)
    plt.show()



# --- test ------------------------------------------------------------------------

if __name__ == '__main__':

    frames = [
        [(-1, 0), (3, 1), 1],
        [(0.5, 1), (2, 0.5), 2],
        [(1, -1), (1, 1), 3],
        [(0.5, 1), (6, 2), 4],
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
    data = np.array(data)
    display(frames, data)