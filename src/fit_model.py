from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split


class FitModel:
    def __init__(self, X, y, scoring):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, random_state=20)

        self.pipeline = Pipeline([('scaler', StandardScaler())])
        self.pipeline.fit(self.X_train, self.y_train)

        self.X_train = self.pipeline.transform(self.X_train)
        self.X_test = self.pipeline.transform(self.X_test)

        self.scoring = scoring

    def find_best_model(self, model, params, n_iter=0):
        if not n_iter:
            grid_search = GridSearchCV(model, params, cv=5, scoring=self.scoring[0],
                                       return_train_score=True, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train.flatten())
            return grid_search

        rnd_search = RandomizedSearchCV(model, param_distributions=params, n_iter=n_iter, cv=5,
                                        scoring=self.scoring[0], random_state=20)
        rnd_search.fit(self.X_train, self.y_train.flatten())

        return rnd_search

    def fit_new_model(self, model, params, name, n_iter=0):
        best_model = self.find_best_model(model, params, n_iter)
        print(f"{name} best model: {best_model.best_params_}")

        y_train_pred = best_model.predict(self.X_train)
        y_test_pred = best_model.predict(self.X_test)

        print(f"{name} {self.scoring[0]}: Train = {self.scoring[1](self.y_train, y_train_pred)} | "
              f"Test = {self.scoring[1](self.y_test, y_test_pred)}")

        return Pipeline([('pipeline', self.pipeline), (name, best_model.best_estimator_)])
