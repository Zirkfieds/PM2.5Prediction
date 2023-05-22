from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def predict_svm(p_train, r_train, C_, gamma_, epsilon_, output):
    # cross-validation under SVM model
    kf = KFold(n_splits=5)
    fold = 0

    prev_train_err, prev_cv_err = 0, 0

    cv_err, train_err = 0, 0

    clf = None

    for train_idx, test_idx in kf.split(p_train):

        fold += 1

        P_train = p_train[train_idx]
        R_train = r_train[train_idx]
        P_test = p_train[test_idx]
        R_test = r_train[test_idx]

        clf = SVR(C=C_, gamma=gamma_, epsilon=epsilon_, kernel='rbf', degree=3, tol=1e-2)
        clf.fit(p_train, r_train)

        r_predict_train = clf.predict(P_train)
        r_predict_test = clf.predict(P_test)

        mse = mean_squared_error(r_predict_train, R_train, squared=False)
        r2 = r2_score(r_predict_train, R_train)

        if output is True:
            print(f'{R_test[:100:20]} | {R_train[:100:20]}')
            print(f'{r_predict_test[:100:20]} | {r_predict_train[:100:20]}')
            print(f'prediction for 83 batch {fold}: {mse}, {r2}')
            # df = pd.DataFrame({'actual': R_train[:1000:10], 'predicted': r_predict_train[:1000:10]})
            # sns.set_style('whitegrid')
            # sns.lineplot(data=df, palette='husl')
            # plt.title('Actual vs. Predicted PM2.5 Concentration')
            # plt.xlabel('Time (hours)')
            # plt.ylabel('PM2.5 Concentration')
            # plt.show()
            # print()

        train_err = mean_squared_error(R_train, r_predict_train)
        train_err = train_err + prev_train_err
        prev_train_err = train_err

        cv_err = mean_squared_error(R_test, r_predict_test)
        cv_err = cv_err + prev_cv_err
        prev_cv_err = cv_err

    train_error = train_err / fold
    cv_error = cv_err / fold

    return clf, train_error, cv_error
