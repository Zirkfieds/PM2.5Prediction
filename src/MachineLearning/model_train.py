import pandas as pd
import numpy as np

from src.MachineLearning.svm import predict_svm
from src.MachineLearning.preprocess import splitter, shuffler
from src.Preprocessing.data_preparation import data_processor
from src.XLSParser.XLSParser import XLSParser


class model_train(object):

    def __init__(self, xls_path, step):
        self.xpath = xls_path
        self.step = step
        self.model = None

    def self_evaluation(self):
        print(self.model)

    def train_model(self):
        xpr = XLSParser(self.xpath)
        processed_data = xpr.preprocess(step=self.step)

        p_train, p_test, r_train, r_test = splitter(processed_data, 0.1)
        p_train, p_test, r_train, r_test = shuffler(p_train, p_test, r_train, r_test)

        # c_range = [1, 5, 10, 50]
        # g_range = [0.005, 0.05, 0.5, 1, 5, 10]
        # e_range = [0.01, 0.1, 0.5, 1, 5]

        c_range = [50]
        g_range = [0.2]
        e_range = [0.01]

        errs = []
        par_c = []
        par_g = []
        par_e = []

        i = 0
        j = 0
        k = 0

        mse_mesh = np.zeros(shape=(len(c_range), len(g_range)))
        te_mesh = np.zeros(shape=(len(c_range), len(g_range)))

        # find the most optimized parameters by traversing all combs of the 2D logspace
        for c in c_range:
            for g in g_range:
                for e in e_range:
                    _, mse, te = predict_svm(p_train, r_train, c, g, 0.01, output=False)
                    print(f'{c} + {g} + {e} + {mse} + {te}')
                    mse_mesh[i][j] = mse
                    te_mesh[i][j] = te
                    errs.append(mse)
                    par_c.append(c)
                    par_g.append(g)
                    par_e.append(e)
                    k += 1
                k = 0
                j += 1
            j = 0
            i += 1

        min_err = errs[0]
        index = -1
        for n in range(len(errs)):
            if errs[n] < min_err:
                min_err = errs[n]
                index = n
        opt_c = par_c[index]
        opt_g = par_g[index]
        opt_e = par_e[index]

        print(f'{opt_c} + {opt_g} + {opt_e}')

        # get the clf model using 美兰站
        clf, _, __, = predict_svm(p_train, r_train, opt_c, opt_g, opt_e, output=True)
        self.model = clf

        return clf
