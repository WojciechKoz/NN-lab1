import matplotlib.pyplot as plt
import numpy as np

def draw_models(models, X, y):
    cols = ['red', 'green']

    for model in models:
        draw_model(model)
    for i, yi in enumerate(np.unique(y)):
        plt.scatter(X[y == yi, 0], X[y == yi, 1], color=cols[i])
    plt.legend()
    plt.xlim(left=-2, right=2)
    plt.ylim(bottom=-2, top=2)
    plt.title("Wizualizacja wyników")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig('plots/adaline_vs_per.png')


def draw_model(model):
    a = -model.w_[0] / model.w_[1]
    b = -model.bias_ / model.w_[1]
    X_start, X_stop = -2, 2
    Y_start = b + a*X_start
    Y_stop = b + a*X_stop

    plt.plot([X_start, X_stop], [Y_start, Y_stop], label=type(model).__name__)
