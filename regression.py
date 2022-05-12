import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.svm import SVR

X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")

print(f"Shape of X: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20)


def linear_regression(X_train, X_test, y_train, y_test):
    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('lr', LinearRegression())])
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print(f"Linear Regression R2 Score: "
          f"Train = {r2_score(y_train, y_train_pred)} | "
          f"Test = {r2_score(y_test, y_test_pred)}")


def ridge(X_train, X_test, y_train, y_test):
    alphas = np.arange(0.01, 15, 0.01)

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('ridge', RidgeCV(alphas=alphas))])
    pipeline.fit(X_train, y_train)

    print("Ridge best alpha:", pipeline['ridge'].alpha_)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print(f"Ridge R2 Score: "
          f"Train = {r2_score(y_train, y_train_pred)} | "
          f"Test = {r2_score(y_test, y_test_pred)}")


def support_vector_regression(X_train, X_test, y_train, y_test):
    alphas = np.arange(0.01, 2, 0.01)
    kernels = np.array(["linear", "poly", "rbf", "sigmoid"])

    best_score = 0
    best_pipeline = None
    best_parameters = {"kernel": None, "alpha": None}

    for kernel in kernels:
        for alpha in alphas:
            pipeline = Pipeline([('scaler', StandardScaler()),
                                 ('svr', SVR(kernel=kernel, C=alpha))])
            # .flatten() to suppress warnings
            pipeline.fit(X_train, y_train.flatten())
            score = pipeline.score(X_test, y_test)
            if score > best_score:
                best_score = score
                best_pipeline = pipeline
                best_parameters["kernel"] = kernel
                best_parameters["alpha"] = alpha

    pipeline = best_pipeline
    print("SVR best kernel:", best_parameters['kernel'])
    print("SVR best alpha:", best_parameters['alpha'])

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print(f"SVR R2 Score: "
          f"Train = {r2_score(y_train, y_train_pred)} | "
          f"Test = {r2_score(y_test, y_test_pred)}")


def elasic_net(X_train, X_test, y_train, y_test):
    l1_ratio = np.arange(0.01, 1.01, 0.01)

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('elastic', ElasticNetCV(l1_ratio=l1_ratio, max_iter=4000))])
    pipeline.fit(X_train, y_train.flatten())

    print("Elastic Net best l1_ratio:", pipeline['elastic'].l1_ratio_)

    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    print(f"Elastic Net R2 Score: "
          f"Train = {r2_score(y_train, y_train_pred)} | "
          f"Test = {r2_score(y_test, y_test_pred)}")


linear_regression(X_train, X_test, y_train, y_test)
ridge(X_train, X_test, y_train, y_test)
support_vector_regression(X_train, X_test, y_train, y_test)
elasic_net(X_train, X_test, y_train, y_test)
