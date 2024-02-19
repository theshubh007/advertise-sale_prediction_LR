import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class DataPreprocessing:
    def __init__(self, df):
        self.df = df

    def identify_outliers(self,data:pd.DataFrame):
        #Identify outliers
        fig, ax = plt.subplots()
        ax.boxplot(data)
        ax.set_title('Boxplot of TV and sales')
        ax.set_ylabel('Value')
        plt.show()
  
    def identify_outliers_zscore(self,data:pd.DataFrame, threshold:float=3):
        #Identify outliers using zscore
        mean = np.mean(data)
        std = np.std(data)
        z_scores = [(i-mean)/std for i in data]
        outliers = np.where(np.abs(z_scores) > threshold)
        return outliers
