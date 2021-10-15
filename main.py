import numpy as np
from perceptron import Perceptron
from adaline import Adaline
import matplotlib.pyplot as plt
from draw_utils import draw_models

SEED = 0
np.random.seed(SEED)

TRIES = 10

X = np.array([[-1, -1], [-1, 1], [1, -1], [1, 1]])
X_uni = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])
y_uni = np.array([0, 0, 0, 1])

X_test = np.array([[1.2, 1.1], [0.1, -0.2], [0, 0], [-0.5, -0.3], [-1, -2]])
y_test = np.array([1, -1, -1, -1, -1])
y_test_uni = np.array([1, 0, 0, 0, 0])


def test_eta(init_model):
    etas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    epochs_lst = []
    for eta in etas:
        epochs_total = 0
        for i in range(TRIES):
            model = init_model(eta=eta)
            epochs_total += model.fit(X, y)
        epochs_lst.append(epochs_total / TRIES)

    print(f"etas {type(init_model()).__name__}")
    print(epochs_lst)
    plt.xscale("log")
    plt.xlabel("Współczynnik uczenia")
    plt.ylabel("Średnia liczba epok")
    plt.title(f"Liczba epok dla różnych wartości współczynnika uczenia w {type(init_model()).__name__}")
    plt.plot(etas, epochs_lst)
    plt.savefig(f"plots/eta_{type(init_model()).__name__}.png")
    plt.close()


def test_perceptron_activation_function():
    using_bipolar = [True, False]
    epochs_lst = []
    for bipolar in using_bipolar:
        correct_y = y if bipolar else y_uni
        correct_X = X if bipolar else X_uni

        epochs_total = 0
        for i in range(TRIES):
            model = Perceptron(bipolar=bipolar)
            epochs_total += model.fit(correct_X, correct_y)
        epochs_lst.append(epochs_total / TRIES)

        print("Using", "Bipolar" if bipolar else "Unipolar")
        print("NO epochs:", epochs_total / TRIES)


def test_adaline_error_threshold():
    thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    epochs_lst = []
    for threshold in thresholds:
        epochs_total = 0
        for i in range(TRIES):
            model = Adaline(eta=0.05, error_threshold=threshold)
            epochs_total += model.fit(X, y)
        epochs_lst.append(epochs_total / TRIES)

    print(f"threshold adaline")
    print(epochs_lst)
    plt.xlabel("Próg błędu")
    plt.ylabel("Epoki")
    plt.title("Liczba epok dla różnych wartości progu błędu")
    plt.plot(thresholds, epochs_lst)
    plt.savefig("plots/adaline_threshold.png")
    plt.close()


def test_init_weights_ranges(init_model):
    ranges = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    epochs_lst = []
    for init_range in ranges:
        epochs_total = 0
        for _ in range(TRIES):
            model = init_model(weights_init_range=init_range)
            epochs_total += model.fit(X, y)
        epochs_lst.append(epochs_total / TRIES)

    print(f"weights {type(init_model()).__name__}")
    print(epochs_lst)
    plt.xscale("log")
    plt.xlabel("Początkowe zakresy wag")
    plt.ylabel("Epoki")
    plt.title(f"Liczba epok dla różnych początkowych wartości wag w {type(init_model()).__name__}")
    plt.plot(ranges, epochs_lst)
    plt.savefig(f"plots/weights_{type(init_model()).__name__}.png")
    plt.close()


def test_init_bias_values(init_model):
    values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5]
    epochs_lst = []
    for value in values:
        epochs_total = 0
        for _ in range(TRIES):
            model = init_model(eta=0.05, weights_init_range=value)
            epochs_total += model.fit(X, y)
        epochs_lst.append(epochs_total / TRIES)

    print(f"bias {type(init_model()).__name__}")
    print(epochs_lst)
    plt.xscale("log")
    plt.xlabel("Początkowe wartości progu")
    plt.ylabel("Epoki")
    plt.title(f"Liczba epok dla różnych wartości progu w {type(init_model()).__name__}")
    plt.plot(values, epochs_lst)
    plt.savefig(f"plots/bias_{type(init_model()).__name__}.png")
    plt.close()


def run_tests():
    test_init_bias_values(Perceptron)
    test_init_bias_values(Adaline)
    test_eta(Perceptron)
    test_eta(Adaline)
    test_perceptron_activation_function()
    test_adaline_error_threshold()
    test_init_weights_ranges(Perceptron)
    test_init_weights_ranges(Adaline)


def run_perceptron():
    model = Perceptron(eta=0.001)
    print("### Perceptron ###")
    print("Epochs:", model.fit(X, y))
    print("train accuracy:", 100 * (model.predict(X) == y).sum() / len(y), "%")
    print("test accuracy:", 100 * (model.predict(X_test) == y_test).sum() / len(y_test), "%")
    print("weights:", model.w_, ", bias:", model.bias_)
    return model


def run_adaline():
    model = Adaline(eta=0.001)
    print("### Adaline ###")
    print("Epochs:", model.fit(X, y))
    print("train accuracy:", 100 * (model.predict(X) == y).sum() / len(y), "%")
    print("test accuracy:", 100 * (model.predict(X_test) == y_test).sum() / len(y_test), "%")
    print("weights:", model.w_, ", bias:", model.bias_)
    return model


models = [run_perceptron(), run_adaline()]
draw_models(models, X, y)
