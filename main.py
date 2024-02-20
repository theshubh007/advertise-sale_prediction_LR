from src.data_ingest import DataIngestion
from src.data_preprocessing import DataPreprocessing
from src.build_model import SingleFeatureLinearRegression
import numpy as np
import time

if __name__ == "__main__":
    start_time = time.time()
    data_ingestion = DataIngestion(data_path="./data/Advertising.csv")

    ### 1. load data and get features and target variable
    X_train, X_test, y_train, y_test, df = data_ingestion.get_X_y()
    df.to_csv("./data/processed_data.csv", index=False)
    # print(df)

    ### 2. pre process data
    data_preprocessing = DataPreprocessing(df)
    outliers = data_preprocessing.identify_outliers_zscore(df["TV"])
    # print(outliers)

    ### 3. build model
    lr_modelC = SingleFeatureLinearRegression(X_train, y_train)
    model = lr_modelC.summaryandtrain()

    ## 4.predict and evaluate
    # lr_model.plot_model(model)
    predicted_value = lr_modelC.predict(model, X_test)
    print(predicted_value)
    # print(type(X_test), type(y_test))
    # evaluation = lr_modelC.evaluate(model, X_test, y_test)
    # print("RMSE: ", evaluation)

    end_time = time.time()
    print(f"Model training took {end_time - start_time} seconds")


