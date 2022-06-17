"""
    Name:       Regression.py
    Authors:    Théo Perinet, Moustapha Diop, Marc Monteil, Mathieu Rivier, Martin Poulard
    Version:    1.0

    This file corresponds to the 4th question of the final FTML project:
    Regression
"""

import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.metrics import r2_score
from sklearn.svm import SVR
from fit_model import FitModel

X = np.load("data/regression/inputs.npy")
y = np.load("data/regression/labels.npy")

print(f"Shape of X: {X.shape}")

if __name__ == '__main__':
    model_regression = FitModel(X, y, ("r2", r2_score))

    lr_pipeline = model_regression.fit_new_model(LinearRegression(), {},
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
                                                  n_iter=True)

    params_en = [
        {"alpha": np.arange(0.01, 2, 0.02),
         "l1_ratio": np.arange(0.01, 1.01, 0.01)}
    ]

    en_pipeline = model_regression.fit_new_model(ElasticNet(random_state=20,
                                                            max_iter=10000),
                                                 params_en, "ElasticNet")
