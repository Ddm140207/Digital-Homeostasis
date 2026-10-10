import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

class BaselineModel:

    def __init__(self):

        self.scaler = StandardScaler()

        self.model = SGDClassifier(loss="log_loss")

        self.initialized = False

    def train_initial(self, X, y):

        Xs = self.scaler.fit_transform(X)

        # Ensure the model sees both classes even if the initial batch is single-class
        classes = np.unique(y)
        if len(classes) == 1:
            classes = np.array([classes[0], 1 - classes[0]])

        self.model.partial_fit(Xs, y, classes=classes)

        self.initialized = True

    def predict(self, X):

        # Ensure the input is a properly shaped numeric array to avoid slow
        # object-dtype conversions inside sklearn's transformer.
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler.transform(X)

        return self.model.predict(Xs)

    def update(self, X, y):

        # baseline still learns but without regulation
        X = np.atleast_2d(np.asarray(X, dtype=float))
        Xs = self.scaler.transform(X)

        self.model.partial_fit(Xs, y)