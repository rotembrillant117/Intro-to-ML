import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


class Perceptron:

    model = None

    def fit(self, X, y):
        y = np.array([y]).T
        phi_zero = np.ones(X.shape[0])
        X = np.insert(X, 0, phi_zero, axis=1)
        w = np.zeros((1, X.shape[1]))
        flag = True
        while flag:
            updated = False
            for i in range(X.shape[0]):
                if y[i] * np.inner(w, X[i]) <= 0:
                    w = w + y[i] * X[i]
                    updated = True
            if updated is False:
                break
        self.model = w
        return

    def predict(self, X):
        phi_zero = np.ones(X.shape[0])
        X = np.insert(X, 0, phi_zero, axis=1)
        y_hat = X @ self.model.transpose()
        y_hat = np.where(y_hat[np.arange(0, X.shape[0])] >= 0, 1, -1)

        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


class LDA:
    pr_y_plus, pr_y_minus, mu_plus, mu_minus, sigma = None, None, None, None, None

    def fit(self, X, y):
        y = np.array([y]).T
        positive = sum([i for i in y if i == 1])
        positive = positive[0]
        negative = y.shape[0] - positive
        self.pr_y_plus = sum([i for i in y if i == 1]) / y.shape[0]
        self.pr_y_minus = 1 - self.pr_y_plus
        self.mu_plus = np.sum(X[np.where(y == 1)[0]], axis=0) * (1 / positive)
        self.mu_minus = np.sum(X[np.where(y == -1)[0]], axis=0) * (1 / negative)
        self.sigma = np.zeros((X.shape[1], X.shape[1]))

        for i in range(X.shape[0]):
            if y[i] == 1:
                self.sigma = self.sigma + np.array([(X[i] - self.mu_plus)]).T @ np.array([(X[i] - self.mu_plus)])
            else:

                self.sigma = self.sigma + np.array([(X[i] - self.mu_minus)]).T @ np.array([(X[i] - self.mu_minus)])
        self.sigma = self.sigma / (X.shape[0] - 2)
        return

    def predict(self, X):

        delta = lambda x, mu, pr_y: x @ np.linalg.inv(self.sigma) @ np.array([mu]).T - \
                                     0.5 * mu @ np.linalg.inv(self.sigma) @ np.array([mu]).T + np.log(pr_y)

        y_hat = np.where(delta(X[np.arange(0, X.shape[0])], self.mu_plus, self.pr_y_plus) >=
                         delta(X[np.arange(0, X.shape[0])], self.mu_minus, self.pr_y_minus), 1, -1)

        return y_hat

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


class SVM:
    svm = SVC(C=1e10, kernel='linear')

    def fit(self, X, y):
        self.svm.fit(X, y)
        return

    def predict(self, X):
        return self.svm.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


class Logistic:
    logistic = LogisticRegression(solver='liblinear')

    def fit(self, X, y):
        self.logistic.fit(X, y)
        return

    def predict(self, X):
        return self.logistic.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


class DecisionTree:
    dec_tree = DecisionTreeClassifier()

    def fit(self, X, y):
        self.dec_tree.fit(X, y)
        return

    def predict(self, X):
        return self.dec_tree.predict(X)

    def score(self, X, y):
        y_hat = self.predict(X)
        return extract_score(X, y, y_hat)


def extract_score(X, y, y_hat):
    y = np.array([y]).T
    score_dic = {}
    P = sum(i for i in y if i == 1)
    P = P[0]
    N = y.shape[0] - P
    y = y.T[0]
    positive = y == 1
    negative = y == -1
    TP = np.count_nonzero(y_hat[positive] == 1)
    FP = np.count_nonzero(y_hat[positive] == -1)
    TN = np.count_nonzero(y_hat[negative] == -1)
    FN = np.count_nonzero(y_hat[positive] == 1)

    score_dic["num samples"] = X.shape[0]
    score_dic["error"] = (FP + FN) / y.shape[0]
    score_dic["accuracy"] = (TP + TN) / y.shape[0]
    score_dic["FPR"] = FP / N
    score_dic["TPR"] = TP / P
    score_dic["precision"] = TP / (TP + FP)
    score_dic["specificty"] = TN / N
    return score_dic
