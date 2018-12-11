from utils import load_mnist
from utils import progress_bar
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer

class Perceptron(object):

    def __init__(self, data, threshold=1000000):
        # Threshold , number of epochs that algorithm will do, default = 100
        self.epochs = threshold

        # Dataframe
        self.data = data

        # TODO: Check if this (len(data[0])) approach is working
        # Number of columns in dataframe
        self.num_of_columns = len(data['data'][0])

        # Inputs (X)
        # Assumption: n-1 columns of data frame contains input data vectors
        # self.X = self.data.iloc[:, :(self.num_of_columns - 1)]
        self.X = []
        for i in range(len(self.data['data'])):
            self.X.append(np.append(self.data['data'][i], [1]))

        # Labels (Y)
        # Assumption: A column n`th of data frame is the labels vector
        # self.Y = self.data.iloc[:, :-1]
        self.Y = []
        for i in range(len(self.data['target'])):
            self.Y.append(1 if self.data['target'][i] == 1 else -1)

        # Weights vector
        # TODO: Calculate right value
        self.weights = np.zeros(self.num_of_columns + 1)

    def train(self):
        # Algorithm implementation
        progress_bar(0, "Training perceptron...")
        for t in range(self.epochs):
            progress_bar(t / self.epochs, "Training perceptron...")
            for i, x in enumerate(self.X):
                if (np.dot(self.X[i], self.weights) * self.Y[i]) <= 0:
                    # TODO: Does it update weights vector at index i, ot should we use self.weights[i]
                    self.weights = self.weights + self.X[i] * self.Y[i]
                    break
            else:
                break
        else:
            progress_bar(1, "Training perceptron completed without convergence")
        print("Perceptron ready")
        return self.weights

    def predict(self, x):
        return 1 if np.dot(np.append(x, [1]), self.weights) > 0 else -1

    def predict_arr(self, X):
        res = []
        for i in range(len(X)):
            res.append(self.predict(X[i]))
        return res


def func(x):
    return 1 + (2*x) + (3 * (x**2))


def grad(x):
    return 2 + (6 * x)


def updt(grad, x, a):
    return x - (a * grad(x))


def ex3a(x_train, y_train):
    x_train = x_train / 255.0 * 2 - 1
    clf = svm.SVC(kernel='linear')
    clf.fit(x_train, y_train)
    return clf


def ex3b(model, x_train, y_train, x_test, y_test):
    x_train = x_train / 255.0 * 2 - 1
    x_test = x_test / 225.0 * 2 - 1
    y_pred_train = model.predict(x_train)
    train_matrix = confusion_matrix(y_train, y_pred_train)
    print(train_matrix)
    y_pred = model.predict(x_test)
    cnf_matrix = confusion_matrix(y_test, y_pred)
    print(cnf_matrix)


def ex4(data):
    perceptron = Perceptron(data)
    w = perceptron.train()
    return w

def ex2():
    # Exercise 2:
    # Show function plot:
    X = np.arange(-5, 5, 0.1)
    Y = [func(x) for x in X]
    plt.plot(X, Y)
    plt.title('1+2x+3x^2')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    # Find minimum:
    EPS = 1e-9
    x = 0
    x_step = updt(grad, x, 0.1)
    while abs(x - x_step) > EPS:
        x = x_step
        x_step = updt(grad, x, 1e-3)
    print("Function minimum is at ({}, {})".format(x_step, func(x_step)))


if __name__ == '__main__':
    ex2()
    # Exercise 3:
    data_df, labels_df = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.143)
    clf = ex3a(X_train, y_train)
    ex3b(clf, X_train, y_train, X_test, y_test)
    ex4(load_breast_cancer())


