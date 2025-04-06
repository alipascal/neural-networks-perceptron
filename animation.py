"""

"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def display(frames, data):

    class UpdateAnimation:
        def __init__(self, ax, data):
            self.success = 0
            self.line, = ax.plot([], [], 'b-')
            self.point = ax.scatter(0, 0, color="red", s=50)
            self.x = np.linspace(-1, 2, 200)
            self.ax = ax

            # plot parameters
            self.ax.set_xlim(-1, 2)
            self.ax.set_ylim(-1, 2)

            N = len(data)
            s = data[:, :2] # set
            c = data[:, 2] # classification
            ax.scatter(s[c == 0, 0], s[c == 0, 1], marker='+', color='black')
            ax.scatter(s[c == 1, 0], s[c == 1, 1], marker='_', color='black')

        def start(self):
            return self.line, self.point

        def __call__(self, frame):
            line, point = frame[0], frame[1]
            a, b = line
            y = a * self.x + b
            self.line.set_data(self.x, y)

            posx, posy = point[0], point[1]
            self.point.set_offsets(np.array([[posx, posy]]))
            return self.line, self.point

    fig, ax = plt.subplots()
    update = UpdateAnimation(ax, data)
    anim = FuncAnimation(fig, update, init_func=update.start, frames=frames, interval=100, blit=True, repeat=False)
    plt.show()


if __name__ == '__main__':
    frames = [
        [(-1, 0), (3, 1)],
        [(0.5, 1), (2, 0.5)],
        [(1, -1), (1, 1)],
        [(0.5, 1), (6, 2)],
    ]
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
    display(frames, data)