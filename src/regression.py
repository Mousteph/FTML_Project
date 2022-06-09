import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV

X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")

print(f"Shape of X: {X.shape}")


class FitModelRegression:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=20)

        self.pipeline = Pipeline([('scaler', StandardScaler())])
        self.pipeline.fit(self.X_train, self.y_train)

        self.X_train = self.pipeline.transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)

    def find_best_model(self, model, params, random=False):
        if not random:
            grid_search = GridSearchCV(
                model, params, cv=5, scoring='r2', return_train_score=True, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train.flatten())
            return grid_search

        rnd_search = RandomizedSearchCV(
            model, param_distributions=params, n_iter=200, cv=5, scoring='r2', random_state=20)
        rnd_search.fit(self.X_train, self.y_train.flatten())

        return rnd_search

    def fit_new_model(self, model, params, name, random=False):
        best_model = self.find_best_model(model, params, random)
        print(f"{name} best model: {best_model.best_params_}")

        y_train_pred = best_model.predict(self.X_train)
        y_test_pred = best_model.predict(self.X_test)

        print(f"{name} R2 Score: Train = {r2_score(self.y_train, y_train_pred)} | "
              f"Test = {r2_score(self.y_test, y_test_pred)}")

        return Pipeline([('pipeline', self.pipeline), (name, best_model.best_estimator_)])


if __name__ == '__main__':
    model_regression = FitModelRegression(X, y)

    lr_pipeline = model_regression.fit_new_model(LinearRegression(), dict(),
                                                 "Linear Regression")

    params_rigde = [
        {"alpha": np.arange(0.01, 5, 0.01),
         "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}
    ]
    ridge_pipeline = model_regression.fit_new_model(Ridge(random_state=20),
                                                    params_rigde, "Ridge")

    params_SVR = [
        {"C": np.arange(0.01, 2, 0.02),
         "kernel": ["linear", "rbf", "sigmoid"],
         "epsilon": np.arange(0.1, 1, 0.1)}
    ]
    svr_pipeline = model_regression.fit_new_model(SVR(), params_SVR, "SVR",
                                                  random=True)

    params_en = [
        {"alpha": np.arange(0.01, 2, 0.02),
         "l1_ratio": np.arange(0.01, 1.01, 0.01)}
    ]

    en_pipeline = model_regression.fit_new_model(ElasticNet(random_state=20,
                                                            max_iter=10000),
                                                 params_en, "ElasticNet")
