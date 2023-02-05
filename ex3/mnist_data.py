from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from models import *
import time


class KNearestNeighbors:

    k_neighbors = KNeighborsClassifier(4)

    def fit(self, X, y):
        self.k_neighbors.fit(X, y)
        return

    def predict(self, X):
        return self.k_neighbors.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


def learner():
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = X_train[train_images], y_train[train_images]
    x_test, y_test = X_test[test_images], y_test[test_images]
    zero_counter = 0
    one_counter = 0
    for i in range(len(y_train)):
        plt.figure()
        if y_train[i] == 1 and one_counter < 3:
            plt.imshow(x_train[i])
            one_counter += 1
        elif y_train[i] == 0 and zero_counter < 3:
            plt.imshow(x_train[i])
            zero_counter += 1
        plt.show()
        if one_counter == 3 and zero_counter == 3:
            break


def get_data(m, train):
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    train_images = np.logical_or((y_train == 0), (y_train == 1))
    test_images = np.logical_or((y_test == 0), (y_test == 1))
    x_train, y_train = X_train[train_images], y_train[train_images]
    x_test, y_test = X_test[test_images], y_test[test_images]
    if train:
        randomizer = np.random.randint(0, high=y_train.shape[0], size=m)
        random_x = x_train[randomizer]
        random_y = y_train[randomizer]
        random_y = np.where(random_y == 0, -1, 1)
        return random_x, random_y

    y_test = np.where(y_test == 0, -1, 1)
    return x_test, y_test


def rearrange_data(X):
    return X.reshape(X.shape[0], -1)


def compare_models_two(row_list):
    logi_accuracy, tree_accuracy, svm_accuracy, neighbors_accuracy = [], [], [], []
    x_test, y_test = get_data(0, False)
    x_test = rearrange_data(x_test)
    for m in row_list:
        logi_counter, tree_counter, svm_counter, neighbors_counter = 0, 0, 0, 0
        logi_runtime, tree_runtime, svm_runtime, neighbors_runtime = 0, 0, 0, 0
        for i in range(50):
            logi = Logistic()
            tree = DecisionTree()
            svm_model = SVM()
            neighbors = KNearestNeighbors()
            flag = True
            while flag:
                X, y = get_data(m, True)
                if len(np.unique(y.T)) == 1:
                    continue
                else:
                    flag = False
            X = rearrange_data(X)
            start_time = time.time()
            logi.fit(X, y)
            logi_counter += logi.score(x_test, y_test)["accuracy"]
            logi_runtime += time.time() - start_time

            start_time = time.time()
            tree.fit(X, y)
            tree_counter += tree.score(x_test, y_test)["accuracy"]
            tree_runtime += time.time() - start_time

            start_time = time.time()
            svm_model.fit(X, y)
            svm_counter += svm_model.score(x_test, y_test)["accuracy"]
            svm_runtime += time.time() - start_time

            start_time = time.time()
            neighbors.fit(X, y)
            neighbors_counter += neighbors.score(x_test, y_test)["accuracy"]
            neighbors_runtime += time.time() - start_time

        print(str(m) + " samples run time:")
        print("Logistic: " + str(logi_runtime))
        print("DecisionTree: " + str(tree_runtime))
        print("Soft SVM: " + str(svm_runtime))
        print("K Nearest Neighbors: " + str(neighbors_runtime))
        print()

        logi_accuracy.append(logi_counter / 50)
        tree_accuracy.append(tree_counter / 50)
        svm_accuracy.append(svm_counter / 50)
        neighbors_accuracy.append(neighbors_counter / 50)

    plt.plot(row_list, tree_accuracy, "purple", label="Decision Tree")
    plt.plot(row_list, logi_accuracy, "green", label="Logistic Regression")
    plt.plot(row_list, svm_accuracy, "red",  label="Soft SVM")
    plt.plot(row_list, neighbors_accuracy, "turquoise", label="K Nearest Neighbors")
    plt.xlabel("m samples")
    plt.ylabel("accuracy mean")
    plt.title("Accuracy of different models depending on amount of m samples")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    row_list = [50, 100, 300, 500]
    learner()
    compare_models_two(row_list)


