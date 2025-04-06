import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def display(frames, data=[]): 

    fig, ax = plt.subplots()
    x_vals = np.linspace(-10, 10, 200)

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    # Traitement des données
    N = len(data)
    plt.scatter(data[:N/2,0], data[:N/2,1], marker='+', label='')
    plt.scatter(data[N/2:,0], data[N/2:,1], marker='_', label='')

    def update(frame):

        for item_type, data in frame:
            if item_type == "line":
                a, b = data
                y = a * x_vals + b
                ax.plot(x_vals, y, color="blue")
            elif item_type == "point":
                x, y = data
                ax.scatter(x, y, color="red", s=50)
        return []

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        repeat=False,
        interval=1000
    )

    plt.show()

frames = [
    [("line", (-1, 0))],
    [("line", (0.5, 1)), ("point", (2, 1))],
    [("line", (1, -1)), ("point", (2, 1))],
    [("line", (0.5, 1)), ("point", (-3, -2))],
    [("point", (-3, -2))]
]

from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
y = iris.target

X = X[:100, [0,2]]

display(frames, X)