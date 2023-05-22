import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, r2_score
from src.MachineLearning.preprocess import splitter, shuffler
from src.Preprocessing.data_preparation import data_processor
from src.XLSParser.XLSParser import XLSParser

import seaborn as sns
import matplotlib.pyplot as plt


class model_test(object):

    def __init__(self, xls_path, step):
        self.xpath = xls_path
        self.step = step

    def single_test(self, clf, step, data):
        xpr = XLSParser(self.xpath)
        test_processed_data = xpr.preprocess(step=step)

        tp_train, tp_test, tr_train, tr_test = splitter(test_processed_data, 0.1)
        tp_train, tp_test, tr_train, tr_test = shuffler(tp_train, tp_test, tr_train, tr_test)

        clf.fit(tp_train, tr_train)

        pred = clf.predict(data)
        print(f'prediction for {data}: {pred}')


    def test_model(self, clf):
        xpr = XLSParser(self.xpath)
        test_processed_data = xpr.preprocess(step=self.step)

        tp_train, tp_test, tr_train, tr_test = splitter(test_processed_data, 0.1)
        tp_train, tp_test, tr_train, tr_test = shuffler(tp_train, tp_test, tr_train, tr_test)

        clf.fit(tp_train, tr_train)

        tr_predict_train = clf.predict(tp_train)

        mse = mean_squared_error(tr_predict_train, tr_train, squared=False)
        r2 = r2_score(tr_predict_train, tr_train)

        # visualization + metrics
        print(f'prediction metrics: {mse}, {r2}')
        df = pd.DataFrame({'actual': tr_train[::5], 'predicted': tr_predict_train[::5]})
        sns.set_style('whitegrid')
        sns.lineplot(data=df, palette='husl')
        plt.title('Actual vs. Predicted PM2.5 Concentration')
        plt.xlabel('Time (hours)')
        plt.ylabel('PM2.5 Concentration')
        plt.show()
