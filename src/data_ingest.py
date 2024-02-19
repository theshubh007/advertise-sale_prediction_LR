import pandas as pd
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_path):
        self.data_path = data_path

    def load_data(self):
        return pd.read_csv(self.data_path)

    def get_X_y(self):
        data=self.load_data()
        x=data[['TV']]
        y=data['sales']
        df=pd.concat([x,y],axis=1)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
        return X_train, X_test, y_train, y_test, df
