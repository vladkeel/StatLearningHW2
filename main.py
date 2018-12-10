from utils import load_mnist
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix

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
    pass


if __name__ == '__main__':
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

    # Exercise 3:
    data_df, labels_df = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.143)
    clf = ex3a(X_train, y_train)
    ex3b(clf, X_train, y_train, X_test, y_test)


