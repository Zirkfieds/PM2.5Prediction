from sklearn.model_selection import train_test_split
import random as rd
import numpy as np


def splitter(data, test_ratio):
    pred = data[:, :-2]
    print(f'{pred.shape} + {pred[0]}')
    resp = data[:, -1:]
    print(f'{resp.shape} + {resp[0]}')
    p_train, p_test, r_train, r_test = train_test_split(pred, resp, test_size=test_ratio, shuffle=False)
    return p_train, p_test, r_train, r_test


def shuffler(p_train, p_test, r_train, r_test, only_ravel=False):

    if not only_ravel:
        indexes = [i for i in range(len(p_train))]

        rd.shuffle(indexes)
        p_train = p_train[indexes]
        r_train = r_train[indexes]
        ind_list = [i for i in range(len(p_test))]
        rd.shuffle(indexes)
        p_test = p_test[ind_list]
        r_test = r_test[ind_list]

    r_train = np.ravel(r_train)
    r_test = np.ravel(r_test)

    return p_train, p_test, r_train, r_test
