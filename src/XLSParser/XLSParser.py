import os
import pandas as pd
import numpy as np

DATASET_PREFIX = '../dataset/'
DATA_FILE = 'data.xls'
TEMP_DATA_FILE = 'tmpxls'

LONGITUDE = 110.335117
LATITUDE = 19.995355

CUTOFF = 1000  # total entries of valid data for PM10

drop_table = ['MN', 'NO2', 'WindDir', 'WindSpd', 'PM10', 'CO', 'Humidity', 'AirPressure', 'Temperature', 'SO2', 'O3']
kept_table = ['Hours', 'PM2.5']

class XLSParser(object):

    def __init__(self, path=DATASET_PREFIX + DATA_FILE):
        self.raw_df = pd.read_excel(path)
        self.df = self.raw_df.copy()
        self.df_shape = self.df.shape

    def preprocess(self, columns_list=None):

        self.df = self.df[:CUTOFF]

        if columns_list is None:
            columns_list = drop_table
        for cols in columns_list:
            self.df = self.df.drop([cols], axis=1)

        lag = 6  # create lags at the length of 6
        for i in range(1, lag + 1):
            self.df[f'PM2.5_{i}'] = self.raw_df['PM2.5'].shift(i)
        self.df['PM2.5_predict'] = self.df['PM2.5'].shift(-1)

        self.df = self.df.dropna(how='any')
        self.df = self.df.drop(['Hours'], axis=1)

        return np.asarray(self.df)
