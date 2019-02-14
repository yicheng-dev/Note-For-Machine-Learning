import csv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

def read_data(filename):
    origin_X = []
    origin_y = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            origin_y.append(row.pop(-1))
            origin_X.append(row)

    origin_X.pop(0)
    origin_y.pop(0)
    return origin_X, origin_y


def Bagging(X, y, ratio, n_trainers):
    clfs = []
    for i in range(0, n_trainers):
        clf = DecisionTreeClassifier()
        x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=1-ratio, train_size=ratio, shuffle=True)
        clf.fit(x_train, y_train)
        clfs.append(clf)
    return clfs


def Bagging_predict(clfs, testX):
    ret = []
    result = []
    for clf in clfs:
        result.append(clf.predict(testX))
    for i in range(0, len(testX)):
        temp_ret = []
        for j in range(0, len(result)):
            temp_ret.append(result[j][i])
        ret.append(max(set(temp_ret), key=temp_ret.count))
    return ret


if __name__ == '__main__':
    X, y = read_data('data_3_0a.csv')
    clfs = Bagging(X, y, 0.8, 10)
    result = Bagging_predict(clfs, X)
