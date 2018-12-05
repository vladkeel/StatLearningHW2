from utils import load_mnist
from sklearn.model_selection import train_test_split


def func(x):
    pass


def grad(x):
    pass


def updt(grad, x, a):
    pass


def ex3a(x_train, y_train):
    pass


def ex3b(model, x_train, y_train, x_test, y_test):
    pass


def ex4(data):
    pass


if __name__ == '__main__':
    data_df, labels_df = load_mnist()
    X_train, X_test, y_train, y_test = train_test_split(data_df, labels_df, random_state=98, test_size=0.143)
