import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def fit_linear_regression(X, y):

    phi_zero = np.ones(X.shape[0])
    more_col_X = np.insert(X, 0, phi_zero, axis=1)
    svd_vals = np.linalg.svd(more_col_X, compute_uv=False)
    X_dagger = np.linalg.pinv(more_col_X)
    w = X_dagger @ y
    return w, svd_vals


def predict(X, w):
    return X @ w


def mse(response_vec, predict_vec):
    lista = []
    for i in range(len(response_vec)):
        lista.append((response_vec[i] - predict_vec[i])**2)
    return sum(lista)/len(response_vec)


def load_data(path):
    data = pd.read_csv(path)
    data = data.dropna()
    del data["long"]
    del data["lat"]
    del data["id"]
    for col in data.columns:
        if col == "date":
            continue
        elif col == "sqft_living":
            data = data.drop(data[data[col] <= 0].index)
        elif col == "bedrooms":
            data = data.drop(data[data[col] < 0.5].index)
        else:
            data = data.drop(data[data[col] < 0].index)

    data["date"] = data["date"].apply(modify_date).astype(float)
    dummies = pd.get_dummies(data["zipcode"])
    data = data.join(dummies)
    del data["zipcode"]
    data = data.reset_index(drop=True)
    return data


def modify_date(x):
    return x[0:4]


def plot_singular_values(singular_vals):

    singular_vals = np.sort(np.array(singular_vals))[::-1]
    plt.figure()
    plt.plot(np.linspace(1, len(singular_vals), len(singular_vals)), singular_vals, 'o', color="black", markersize=2)
    plt.xlabel("singular value number")
    plt.ylabel("value of singular value")
    plt.title("scree-plot")
    plt.show()


def plot_data(path):
    processed_data = load_data(path)
    price_data = [processed_data["price"]]
    y = pd.concat(price_data, axis=1, keys="price")
    del processed_data["price"]
    processed_data = processed_data.reset_index(drop=True)
    w, svd_vals = fit_linear_regression(processed_data.to_numpy(), y.to_numpy())
    plot_singular_values(svd_vals)
    return


def train_test_lr(path):

    data = load_data(path)
    train = data.sample(frac=0.75)
    test = data.drop(train.index)

    price_data = [test["price"]]
    y = pd.concat(price_data, axis=1, keys="price")

    test = test.drop(["price"], axis=1)
    phi_zero = np.ones(test.shape[0])
    test.insert(0, "phi_zero", phi_zero)

    prediction_list = []
    for i in range(1, 101):
        temp_data_i = train.head(int((i / 100) * train.shape[0]))
        price_data = [temp_data_i["price"]]
        train_y = pd.concat(price_data, axis=1, keys="price")
        w, svd_vals = fit_linear_regression(temp_data_i.drop(["price"], axis=1).to_numpy(), train_y.to_numpy())
        prediction_list.append(mse(y.to_numpy(), predict(test.to_numpy(), w)))

    plt.figure()
    plt.plot(prediction_list, "b")
    plt.xlabel("percentage of train size")
    plt.ylabel("MSE")
    plt.title("Result of MSE as a dependence on test size")
    plt.show()


def feature_evaluation(data, response_vector):

    for i in range(0, 16):
        plt.figure()
        plt.scatter(data[data.columns[i]], response_vector)
        plt.xlabel(str(data.columns[i]) + " Values")
        plt.ylabel("Response Vector Values")
        pearson_corl = np.cov(data[data.columns[i]].to_numpy(), response_vector.to_numpy()) / \
                       (np.std(data[data.columns[i]].to_numpy()) * np.std(response_vector.to_numpy()))

        plt.title("Correlation Between " + str(data.columns[i]) + " Values and Response Values\nPearson Correlation: "
                  + str(pearson_corl[0, 1]))
        plt.show()


if __name__ == '__main__':
    data = load_data("kc_house_data.csv")
    feature_evaluation(data.drop("price", axis=1), data["price"])
    plot_data("kc_house_data.csv")
    train_test_lr("kc_house_data.csv")

