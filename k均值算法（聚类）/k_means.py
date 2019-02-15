import csv
import random
import time
import numpy as np


def read_data(filename):
    origin_X = []
    with open(filename) as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            origin_X.append(row)

    origin_X.pop(0)
    return np.array(origin_X).astype('float64')


def k_means(X, k, max_T):
    mu = random.sample(X.tolist(), k)
    count = 0
    while True:
        update_flag = False
        clusters = []
        d = []
        lamda = []
        for i in range(0, k):
            clusters.append([])
        for j in range(0, len(X)):
            d1 = []
            for i in range(0, k):
                sum_d = np.sum(np.square(X[j] - np.array(mu[i])))
                d1.append(np.sqrt(sum_d))
            d.append(d1)
            lamda.append(d[j].index(min(d[j])))
            clusters[lamda[j]].append(X[j])
        for i in range(0, k):
            new_mu_i = np.divide(np.sum(clusters[i], axis=0), len(clusters[i]))
            new_mu_i = new_mu_i.tolist()
            if not new_mu_i == mu[i]:
                mu[i] = new_mu_i
                update_flag = True
        count += 1
        if count >= max_T or not update_flag:
            break
    return clusters


if __name__ == '__main__':
    random.seed(time.time())
    X = read_data('data_4_0.csv')
    clusters = k_means(X, 3, 10)
    print(clusters)