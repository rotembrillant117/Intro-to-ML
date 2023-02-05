import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn import linear_model
from sklearn import metrics


def Q4(deviation):
    """
    Does the k-fold algorithm and plots the results
    :param deviation: a given deviation
    :return:
    """
    x = np.random.uniform(-3.2, 2.2, size=1500)
    func = (x+3)*(x+2)*(x+1)*(x-1)*(x-2)
    epsilon = np.random.normal(0, deviation, size=1500)
    y = func + epsilon
    index_train = np.random.choice(np.arange(0,1500), size=1000, replace=False)
    index_test = np.delete(np.arange(0,1500), index_train)
    x_train = x[index_train]
    y_train = y[index_train]
    x_test = x[index_test]
    y_test = y[index_test]

    ###################################################################
    S = np.array([x_train[0:500], y_train[0:500]])
    V = np.array([x_train[500:], y_train[500:]])
    two_fold_models = []
    mse_loss = []
    for d in range(1, 16):
        two_fold_models.append(np.poly1d(np.polyfit(S[0], S[1], d)))
    for p in two_fold_models:
        mse_loss.append(np.sum(((p(V[0]) - V[1])**2)) / 500)
    h_index = np.argmin(np.asarray(mse_loss))
    h_hat = two_fold_models[h_index]
    ###################################################################


    k_fold_sets = []
    my_range = np.arange(0, 1000)
    for i in range(0, 5):
        new_set = np.random.choice(my_range, size=200, replace=False)
        k_fold_sets.append(np.array([ x[index_train[new_set]], y[index_train[new_set]] ]))
        my_range = np.delete(my_range, np.in1d(my_range, new_set))

    all_train_error = []
    all_validation_error = []
    for d in range(1, 16):
        train_error_d = []
        validation_error_d = []
        for i in range(0, 5):
            train_sets = np.asarray(k_fold_sets)[np.delete(np.arange(0, 5), i)]
            validation_set = k_fold_sets[i]
            train_sets_unified = np.concatenate([t for t in train_sets], axis=1)
            my_poly = np.poly1d(np.polyfit(train_sets_unified[0], train_sets_unified[1], d))
            train_error_d.append(np.sum(((my_poly(train_sets_unified[0]) - train_sets_unified[1])**2)) / 800)
            validation_error_d.append(np.sum(((my_poly(validation_set[0]) - validation_set[1])**2)) / 200)

        all_train_error.append(np.sum(np.asarray(train_error_d)) / 5)
        all_validation_error.append(np.sum(np.asarray(validation_error_d)) / 5)

    plt.plot(np.arange(1, 16), all_train_error, "blue", label="Train error")
    plt.plot(np.arange(1, 16), all_validation_error, "purple", label="Validation error")
    plt.xlabel("degree of polynomial")
    plt.ylabel("error rate")
    plt.title("Training and Validation error with deviation:" + str(deviation))
    plt.legend()
    plt.show()

    h_star_d = np.argmin(np.asarray(all_validation_error))
    print("The validation error is: " + str(all_validation_error[h_star_d]))
    h_star = np.poly1d(np.polyfit(x_train, y_train, h_star_d + 1))
    test_error = np.sum(((h_star(x_test) - y_test)**2)) / 500
    print("The degree of the polynomial with the lowest test error for deviation " + str(deviation)
          + " is " + str(h_star_d + 1))
    print("test error for deviation " + str(deviation) + ": " + str(test_error))


def Q5():
    """
    Does the k-fold algorithm for the lasso and ridge algorithms and plots the results
    :return:
    """
    X, y = datasets.load_diabetes(return_X_y=True)
    x_train = X[0:50]
    y_train = y[0:50]
    kf = KFold(5, shuffle=True)
    lam = np.linspace(0.01, 2, num=100)
    k_fold_sets = []
    for t, v in kf.split(x_train):
        k_fold_sets.append((t, v))
    all_ridge_train_error = []
    all_lasso_train_error = []
    all_ridge_validation_error = []
    all_lasso_validation_error = []
    for alpha in lam:
        ridge_train_error = []
        lasso_train_error = []
        ridge_validation_error = []
        lasso_validation_error = []
        for train_index, validation_index in k_fold_sets:
            ridge = linear_model.Ridge(alpha)
            lasso = linear_model.Lasso(alpha)
            ridge.fit(x_train[train_index], y_train[train_index])
            lasso.fit(x_train[train_index], y_train[train_index])

            ridge_train_error.append(metrics.mean_squared_error(y_train[train_index],
                                                                ridge.predict(x_train[train_index])))
            lasso_train_error.append(metrics.mean_squared_error(y_train[train_index],
                                                                lasso.predict(x_train[train_index])))

            ridge_validation_error.append(metrics.mean_squared_error(y_train[validation_index],
                                                                     ridge.predict(x_train[validation_index])))
            lasso_validation_error.append(metrics.mean_squared_error(y_train[validation_index],
                                                                     lasso.predict(x_train[validation_index])))

        all_ridge_train_error.append(np.sum(np.asarray(ridge_train_error)) / 5)
        all_ridge_validation_error.append(np.sum(np.asarray(ridge_validation_error)) / 5)

        all_lasso_train_error.append(np.sum(np.asarray(lasso_train_error)) / 5)
        all_lasso_validation_error.append(np.sum(np.asarray(lasso_validation_error)) / 5)

    plt.plot(lam, all_ridge_train_error, "blue", label="Ridge train error")
    plt.plot(lam, all_ridge_validation_error, "red", label="Ridge validation error")
    plt.plot(lam, all_lasso_train_error, "green", label="Lasso train error")
    plt.plot(lam, all_lasso_validation_error, "purple", label="Lasso validation error")
    plt.xlabel("lambda")
    plt.ylabel("error rate")
    plt.title("Train and Validation errors for Ridge and Lasso")
    plt.legend()
    plt.show()

    best_lambda_ridge = np.argmin(np.asarray(all_ridge_validation_error))
    best_lambda_lasso = np.argmin(np.asarray(all_lasso_validation_error))
    print("Best lambda for Ridge is " + str(lam[best_lambda_ridge]))
    print("Best lambda for Lasso is " + str(lam[best_lambda_lasso]))

    best_ridge_hyp = linear_model.Ridge(lam[best_lambda_ridge]).fit(x_train, y_train)
    best_lasso_hyp = linear_model.Lasso(lam[best_lambda_lasso]).fit(x_train, y_train)
    linear_reg = linear_model.LinearRegression().fit(x_train, y_train)
    print("Test error for best Ridge hypothesis " +
          str(metrics.mean_squared_error(best_ridge_hyp.predict(X[50:]), y[50:])))
    print("Test error for best Lasso hypothesis " +
          str(metrics.mean_squared_error(best_lasso_hyp.predict(X[50:]), y[50:])))
    print("Test error for best Linear Regression " +
          str(metrics.mean_squared_error(linear_reg.predict(X[50:]), y[50:])))


if __name__ == '__main__':

    Q4(1)
    Q4(5)
    Q5()
