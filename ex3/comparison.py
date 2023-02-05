import numpy as np
import matplotlib.pyplot as plt
from models import *


def draw_points(m):
    flag = True
    X = np.random.multivariate_normal([0, 0], np.identity(2), size=m)
    y = np.zeros((X.shape[0], 1))
    f = lambda k, x: np.inner(k, x) + 0.1
    for i in range(m):
        if f(np.array([0.3, -0.5]), X[i]) >= 0:
            y[i] = 1
        else:
            y[i] = -1
    return X, y


def plot_points(row_list):

    for m in row_list:
        flag = True
        while flag:
            X, y = draw_points(m)
            if len(np.unique(y.T)) == 1:
                continue
            else:
                flag = False
        plt.figure()
        plus_coord = X[np.where(y == 1)[0]]
        minus_coord = X[np.where(y == -1)[0]]
        x = np.linspace(-5,5,100)
        y_f = x * 0.6 + 0.2
        perce = Perceptron()
        svm_model = SVM()
        svm_model.fit(X, y.T[0])
        perce.fit(X, y.T[0])
        w = svm_model.svm.coef_[0]
        y_perce = x * (- perce.model[0][1]/perce.model[0][2]) - perce.model[0][0] / perce.model[0][2]
        y_svm = x * (- w[0]/w[1]) - svm_model.svm.intercept_[0] / w[1]

        plt.plot(x, y_f, "purple", linewidth=0.6, label="True Hypothesis")
        plt.plot(x, y_perce, "turquoise", linewidth=0.6, label="Perceptron")
        plt.plot(x, y_svm, "red", linewidth=0.6, label="SVM")
        plt.scatter(plus_coord[:, 0], plus_coord[:, 1], c='blue', s=10)
        plt.scatter(minus_coord[:, 0], minus_coord[:, 1], c='orange', s=10)
        plt.title("Classification of m = " + str(m) + " samples")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.show()


def compare_models(row_list):
    perce_accuracy = []
    lda_accuracy = []
    svm_accuracy = []
    for m in row_list:
        perce_counter = 0
        lda_counter = 0
        svm_counter = 0
        for i in range(500):
            perce = Perceptron()
            lda = LDA()
            svm_model = SVM()
            flag = True
            while flag:
                X, y = draw_points(m)
                if len(np.unique(y.T)) == 1:
                    continue
                else:
                    flag = False
            perce.fit(X, y.T[0])
            lda.fit(X, y.T[0])
            svm_model.fit(X, y.T[0])

            X_test, y_test = draw_points(10000)
            perce_counter += perce.score(X_test, y_test.T[0])["accuracy"]
            lda_counter += lda.score(X_test, y_test.T[0])["accuracy"]
            svm_counter += svm_model.score(X_test, y_test.T[0])["accuracy"]
        perce_accuracy.append(perce_counter / 500)
        lda_accuracy.append(lda_counter / 500)
        svm_accuracy.append(svm_counter / 500)

    plt.plot(row_list, lda_accuracy, "purple", label="LDA")
    plt.plot(row_list, perce_accuracy, "turquoise", label="Perceptron")
    plt.plot(row_list, svm_accuracy, "red", label="SVM")
    plt.xlabel("m samples")
    plt.ylabel("accuracy mean")
    plt.title("Accuracy of different models depending on amount of m samples")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    row_list = [5, 10, 15, 25, 70]
    plot_points(row_list)
    compare_models(row_list)
