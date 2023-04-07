import XLSParser.XLSParser as xp
from MachineLearning.preprocess import splitter, shuffler
from MachineLearning.linear_models import predict_svm

import numpy as np

if __name__ == '__main__':
    print('PM 2.5 Prediction')
    xpr = xp.XLSParser()
    processed_data = xpr.preprocess()

    p_train, p_test, r_train, r_test = splitter(processed_data, 0.1)
    p_train, p_test, r_train, r_test = shuffler(p_train, p_test, r_train, r_test)

    # c_range = np.logspace(-1, 1, 3)
    # g_range = np.logspace(-2, 0, 5)

    c_range = [12.0]
    g_range = [0.075]

    errs = []
    par_c = []
    par_g = []

    i = 0
    j = 0

    mse_mesh = np.zeros(shape=(len(c_range), len(g_range)))
    te_mesh = np.zeros(shape=(len(c_range), len(g_range)))

    for c in c_range:
        for g in g_range:
            mse, te = predict_svm(p_train, r_train, c, g, 0.01, output=False)
            print(c, g, mse, te)
            mse_mesh[i][j] = mse
            te_mesh[i][j] = te
            errs.append(mse)
            par_c.append(c)
            par_g.append(g)
            j += 1
        j = 0
        i += 1

    min_err = errs[0]
    index = -1
    for k in range(len(errs)):
        if errs[k] < min_err:
            min_err = errs[k]
            index = k
    opt_c = par_c[index]
    opt_g = par_g[index]

    _, __, = predict_svm(p_train, r_train, opt_c, opt_g, 0.01, output=True)
