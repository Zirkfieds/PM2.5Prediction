import pandas as pd
import numpy as np
import warnings

drop_table = ['MN', 'NO2', 'WindDir', 'WindSpd', 'PM10', 'CO', 'Humidity', 'AirPressure', 'Temperature', 'SO2', 'O3']
kept_table = ['Hours', 'PM2.5']

warnings.simplefilter(action='ignore', category=FutureWarning)

class data_processor(object):

    def __init__(self, xls_df):
        # self.data = pd.read_excel(xls_path)
        self.data = xls_df

    def clean(self):

        means = self.data.mean(numeric_only=True)
        stds = self.data.std(numeric_only=True)

        error_val = ((self.data - means).abs() > 6 * stds).any(axis=1)
        print(f'found {len(self.data) - len(error_val)} error values')
        self.data = self.data[~error_val]

        size_before = len(self.data)

        # optional dropouts
        # self.data = self.data[(self.data['CO'] <= 50) & (self.data['CO'] >= 0)]
        # self.data = self.data[(self.data['SO2'] <= 500) & (self.data['SO2'] >= 0)]
        # self.data = self.data[(self.data['O3'] <= 500) & (self.data['O3'] >= 0)]
        #
        # print(f'found {size_before - len(self.data)} unexpected values')
        # size_before = len(self.data)

        self.data = self.data.dropna()
        print(f'found {size_before - len(self.data)} na values')

    def get_data(self):
        return self.data

    def PCA(self):



        pass

    def write(self):
        pass

    def load(self):
        pass
