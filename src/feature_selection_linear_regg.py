import pandas as pd
from src import config

df1 = pd.read_csv(config.data_processed)
corrmatrix=df.corr()


def correlatedfeature(corr_data, threshold):
    """Function to get features which are correlated to target variable with a particulat threshold"""
    feature = []
    value = []

    for i, index in enumerate(corr_data.index):
        if abs(corr_data[index]) > threshold:
            feature.append(index)
            value.append(corr_data[index])

    df = pd.DataFrame(data=value, index=feature, columns=['corr_value'])
    return df

threshold=0.50
corrvalue=correlatedfeature(corrmatrix['MEDV'], threshold)
correlateddata=df[corrvalue.index]
