import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np


class SingleFeatureLinearRegression:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def train_model(self):
        model = LinearRegression()
        model.fit(self.x, self.y)
        return model

    def summaryandtrain(self):
        model = self.train_model()
        print("Intercept: ", model.intercept_)
        print("Coefficient: ", model.coef_)
        print("Rank of matrix X: ", model.rank_)
        print("Singular values of X: ", model.singular_)
        print("Independent term in the linear model: ", model.intercept_)
        return model

    def plot_model(self, model):
        plt.scatter(self.x, self.y, color="black")
        plt.plot(self.x, model.predict(self.x), color="blue")
        plt.title("Linear Regression")
        plt.xlabel("TV")
        plt.ylabel("Sales")
        plt.show()

    def predict(self, model, x):
        y_pred = model.predict(x)
        return y_pred

    def evaluate(self, model, x, y):
        y_pred = model.predict(x)
        mse = np.mean((y - y_pred) ** 2)
        rmse = np.sqrt(mse)
        return rmse
