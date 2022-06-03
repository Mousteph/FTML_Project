"""
    Name:       Classification.py
    Authors:    Théo Perinet, Moustapha Diop, Marc Monteil, Mathieu Rivier
    Version:    1.0

    This file corresponds to the 5th question of the final FTML project:
    Classification
"""

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


class Classification(object):
    def __init__(self):
        self.X = np.load("data/classification/inputs.npy")
        self.y = np.load("data/classification/labels.npy")

        print(f'Shape of X: {self.X.shape}')
        self.X_train, self.X_test, self.y_train, self.y_test = self._train_test_split(
        )

    def logistic_regression(self):
        model = LogisticRegression(random_state=0)
        model.fit(self.X_train, self.y_train.ravel())
        return model

    def get_perfs(self):
        models = [self.support_vector_machine(), self.logistic_regression()]

        accuracies = []
        for model in models:
            accuracies.append(self.run_algo(model))

        for i in range(len(accuracies)):
            # To fix : Printing the right name
            print(f'Model {type(models[i]).__name__}:')
            print(f'accuracy: {accuracies[i] * 100}%\n')

    def run_algo(self, model):
        y_pred = model.predict(self.X_test)
        return self._test_acuracy(y_pred)

    def support_vector_machine(self):
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        clf.fit(self.X_train, self.y_train.ravel())
        return clf

    def k_nearest_neighbours(self):
        pass

    def naive_bais(self):
        pass

    def decision_tree(self):
        pass

    def _train_test_split(self):
        return train_test_split(self.X, self.y, random_state=42)

    def _test_acuracy(self, y_pred):
        # With normalisation (normalize = False)
        y_test = self.y_test
        return accuracy_score(y_test, y_pred)    #, normalize=False)


if __name__ == "__main__":
    classification = Classification()
    # SVM = classification.support_vector_machine()
    # LR = classification.logistic_regression()
    # print(classification.run_algo(SVM))
    classification.get_perfs()
