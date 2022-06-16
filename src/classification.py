"""
    Name:       Classification.py
    Authors:    Théo Perinet, Moustapha Diop, Marc Monteil, Mathieu Rivier
    Version:    1.0

    This file corresponds to the 5th question of the final FTML project:
    Classification
"""

import numpy as np

from sklearn.linear_model import LogisticRegression, RidgeClassifier, Perceptron
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

from fit_model import FitModel


X = np.load("data/classification/inputs.npy")
y = np.load("data/classification/labels.npy")

print(f"Shape of X: {X.shape}")

if __name__ == '__main__':
    model_regression = FitModel(X, y, ("accuracy", accuracy_score))

    params_lr = [
        {"penalty": ["l1", "l2"],
         "C": np.arange(0.01, 3, 0.01),
         "solver": ["liblinear"]},

        {"penalty": ["l2"],
         "C": np.arange(0.01, 3, 0.01),
         "solver": ["sag"]},

        {"penalty": ["l1", "l2"],
         "C": np.arange(0.01, 3, 0.01),
         "solver": ["saga"]},

        {"penalty": ["elasticnet"],
         "C": np.arange(0.01, 3, 0.01),
         "l1_ratio": np.arange(0, 1, 0.01),
         "solver": ["saga"]},
    ]

    lr_pipeline = model_regression.fit_new_model(LogisticRegression(random_state=20, max_iter=2000),
                                                 params_lr, "Logistic Regression", n_iter=10)

    params_svc = [
        {"C": np.arange(0.01, 3, 0.01),
         "kernel": ["linear", "poly", "rbf", "sigmoid"]}
    ]

    svc_pipeline = model_regression.fit_new_model(
        SVC(random_state=20), params_svc, "SVC")

    params_perceptron = [
        {"penalty": ["elasticnet"],
         "alpha": np.arange(0.001, 0.1, 0.001),
         "l1_ratio": np.arange(0.01, 1, 0.01),
         "eta0": np.arange(0.01, 2.5, 0.01),
         }
    ]

    perceptron_pipeline = model_regression.fit_new_model(Perceptron(random_state=20, n_jobs=-1), params_perceptron,
                                                         "Perceptron", n_iter=6000)

    params_ridge = [
        {"alpha": np.arange(0.01, 4, 0.01),
         "solver": ["svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"],
         }
    ]

    rfc_pipeline = model_regression.fit_new_model(
        RidgeClassifier(random_state=20), params_ridge, "Ridge Classifier")
