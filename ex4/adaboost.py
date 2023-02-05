"""
===================================================
     Introduction to Machine Learning (67577)
===================================================

Skeleton for the AdaBoost classifier.

"""
import numpy as np
from ex4_tools import *


class AdaBoost(object):

    def __init__(self, WL, T):
        """
        Parameters
        ----------
        WL : the class of the base weak learner
        T : the number of base learners to learn
        """
        self.WL = WL
        self.T = T
        self.h = [None]*T     # list of base learners
        self.w = np.zeros(T)  # weights
        self.final_D = None

    def train(self, X, y):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        Train this classifier over the sample (X,y)
        After finish the training return the weights of the samples in the last iteration.
        """
        D_t = np.full(X.shape[0], 1 / X.shape[0])

        for i in range(self.T):
            self.final_D = D_t
            learner = self.WL(D_t, X, y)
            y_hat = learner.predict(X)
            diff = np.sum(D_t[np.where(np.where(y_hat != y, 1, 0) == 1)[0]])
            w_t = 0.5 * np.log((1 / diff) - 1)
            upper = D_t * np.exp(y * y_hat * (-w_t))
            under = np.sum(D_t * np.exp(y * y_hat * (-w_t)))
            D_t_next = upper / under
            self.h[i] = learner
            self.w[i] = w_t
            D_t = D_t_next
        return self.final_D

    def predict(self, X, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: y_hat : a prediction vector for X. shape=(num_samples)
        Predict only with max_t weak learners,
        """
        y_hats = np.zeros((max_t, X.shape[0]))
        for i in range(max_t):
            y_hats[i] = self.h[i].predict(X) * self.w[i]
        y_hat = np.sign(np.sum(y_hats, axis=0))
        y_hat[y_hat == 0] = -1
        return y_hat

    def error(self, X, y, max_t):
        """
        Parameters
        ----------
        X : samples, shape=(num_samples, num_features)
        y : labels, shape=(num_samples)
        :param max_t: integer < self.T: the number of classifiers to use for the classification
        :return: error : the ratio of the wrong predictions when predict only with max_t weak learners (float)
        """
        y_hat = self.predict(X, max_t)
        ratio = np.sum(np.where(y_hat != y, 1, 0)) / y.shape[0]
        return ratio


def plot_T_learners(noise):
    """
    This function plots the decision boundaries and error rates for T learners with given noise to be added to
    train and testing data
    :param noise: the noise to add
    :return:
    """

    x_train, y_train = generate_data(5000, noise)
    x_test, y_test = generate_data(200, noise)
    booster = AdaBoost(DecisionStump, 500)
    booster.train(x_train, y_train)
    train_error, test_error = [], []
    for i in range(1, 501):
        train_error.append(booster.error(x_train, y_train, i))
        test_error.append(booster.error(x_test, y_test, i))
    x = np.arange(1, 501)
    plt.plot(x, train_error, "purple", label="train error")
    plt.plot(x, test_error, "red", label="test error")
    plt.xlabel("T stumps")
    plt.ylabel("error percentage")
    plt.title("Error percentages as a function of T with noise = " + str(noise))
    plt.legend()
    plt.show()
    T, error_list = [5, 10, 50, 100, 200, 500], []
    index = 0
    for t in range(1, 501):
        if t in T:
            plt.subplot(3, 2, index + 1)
            decision_boundaries(booster, x_test, y_test, t)
            plt.title("Decision boundary T = " + str(t) + " for Noise = " + str(noise), fontsize=8)
            index += 1
        error_list.append(booster.error(x_test, y_test, t))
    plt.show()
    min_T_hat = error_list.index(min(error_list))
    lowest_error = booster.error(x_test, y_test, min_T_hat)
    plt.figure()
    decision_boundaries(booster, x_train, y_train, min_T_hat, 1)
    plt.title("Best T = " + str(min_T_hat + 1) + " for Noise = " + str(noise) + " with Error = " + str(lowest_error))
    plt.show()

    plt.figure()
    booster.final_D = booster.final_D / np.max(booster.final_D) * 10
    decision_boundaries(booster, x_train, y_train, 500, booster.final_D)
    plt.title("Decision boundary for T = 500, Noise = " + str(noise))
    plt.show()


if __name__ == '__main__':
    noise = [0, 0.01, 0.4]
    for n in noise:
        plot_T_learners(n)

