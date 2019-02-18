import csv
import numpy as np


class Sample:
    def __init__(self, id, data):
        self.id = id
        self.data = data

    def __str__(self):
        return 'id: %s\tdata: %s' % (self.id, self.data)


def read_data(filename, regression):
    origin_X = []
    origin_y = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        i = 0
        for row in csv_reader:
            if i == 0:
                i += 1
                continue
            if regression:
                origin_y.append(Sample(i, np.array(row.pop(-1)).astype('float64')))
            else:
                origin_y.append(Sample(i, row.pop(-1)))
            origin_X.append(Sample(i, np.array(row).astype('float64')))
            i += 1
    return origin_X, origin_y


def dist(xi, xj):
    return np.sqrt(np.sum(np.square(xi - xj)))


def kNN(X, y, testX, k, regression):
    X.sort(key=lambda x: dist(x.data, testX), reverse=False)
    if regression:
        sum = 0
        for i in range(0, k):
            sum += y[X[i].id - 1]
        return sum / k
    else:
        vote_pool = []
        for i in range(0, k):
            vote_pool.append(y[X[i].id - 1].data)
        return max(set(vote_pool), key=vote_pool.count)


if __name__ == '__main__':
    X, y = read_data('data_3_0a.csv', regression=False)
    result = kNN(X=X, y=y, testX=np.array([0.5, 0.5]), k=3, regression=False)
    print(result)
