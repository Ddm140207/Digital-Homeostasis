from collections import deque
import numpy as np


class BaselineSystem:

    def __init__(self, model, batch_size=32):

        self.model = model

        self.correct = 0
        self.total = 0

        self.errors = []

        self.batch_size = batch_size
        self.buffer_X = deque()
        self.buffer_y = deque()

    def step(self, X, y):

        pred = self.model.predict([X])[0]

        if pred == y:
            self.correct += 1

        self.total += 1

        error = int(pred != y)

        self.errors.append(error)

        # store sample
        self.buffer_X.append(X)
        self.buffer_y.append(y)

        # update only every batch
        if len(self.buffer_X) >= self.batch_size:

            X_batch = np.array(self.buffer_X)
            y_batch = np.array(self.buffer_y)

            self.model.update(X_batch, y_batch)

            self.buffer_X.clear()
            self.buffer_y.clear()

    def accuracy(self):

        return self.correct / max(self.total, 1)