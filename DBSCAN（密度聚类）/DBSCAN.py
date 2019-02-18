import csv
import random
import time
import numpy as np
import queue


class Sample:
    def __init__(self, id, data):
        self.id = id
        self.data = data

    def __str__(self):
        return 'id: %s\tdata: %s' % (self.id, self.data)



def read_data(filename):
    origin_X = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            origin_X.append(row)

    origin_X.pop(0)
    return np.array(origin_X).astype('float64')


def dist(xi, xj):
    return np.sqrt(np.sum(np.square(xi - xj)))


def DBSCAN(X, epsilon, MinPts):
    Omega = []
    nbhood = []
    for j in range(0, len(X)):
        nbhood_j = []
        for i in range(0, len(X)):
            if dist(X[i], X[j]) < epsilon:
                nbhood_j.append(X[i])
        nbhood.append(Sample(j, nbhood_j))
        if len(nbhood_j) >= MinPts:
            Omega.append(Sample(j, X[j]))
    k = 0
    clusters = []
    Gamma = []
    for i in range(0, len(X)):
        Gamma.append(Sample(i, X[i]))
    while not len(Omega) == 0:
        Gamma_old = Gamma
        omega = np.random.choice(Omega)
        Q = queue.Queue()
        Q.put(omega)
        for i in range(0, len(Gamma)):
            if (Gamma[i].data == np.array(omega.data)).all():
                Gamma = np.delete(Gamma, i, axis=0)
                break
        while not Q.empty():
            q = Q.get()
            if len(nbhood[q.id].data) >= MinPts:
                Delta = []
                for data in nbhood[q.id].data:
                    for gamma in Gamma:
                        if (gamma.data == np.array(data)).all():
                            Delta.append(gamma)
                for delta in Delta:
                    Q.put(delta)
                    for i in range(0, len(Gamma)):
                        if (Gamma[i].data == np.array(delta.data)).all():
                            Gamma = np.delete(Gamma, i, axis=0)
                            break
        k += 1
        cluster = []
        for gamma_old in Gamma_old:
            if gamma_old not in Gamma:
                cluster.append(gamma_old)
        clusters.append(cluster)
        Omega_new = []
        for omega in Omega:
            append_flag = True
            for c in cluster:
                if (np.array(c.data) == omega.data).all():
                    append_flag = False
                    break
            if append_flag:
                Omega_new.append(omega)
        Omega = Omega_new
        # print(k)
    return clusters



if __name__ == '__main__':
    random.seed(time.time())
    X = read_data('data_4_0.csv')
    clusters = DBSCAN(X, epsilon=0.11, MinPts=5)
    for cluster in clusters:
        for sample in cluster:
            print(sample.id, ' ', end='')
        print()
