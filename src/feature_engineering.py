import pandas as pd
from src import config

df = pd.read_csv(config.data_path)

df.CRIM.fillna(df['CRIM'].mean(),inplace=True)
df.ZN.fillna(df['ZN'].mean(),inplace=True)
df.INDUS.fillna(df['INDUS'].mean(),inplace=True)
df.CHAS.fillna(df['CHAS'].mean(),inplace=True)
df.AGE.fillna(df['AGE'].mean(),inplace=True)
df.LSTAT.fillna(df['LSTAT'].mean(),inplace=True)
#print(df.info())

df.to_csv('C:/Users/Admin/PycharmProjects/pythonProject3/house_price/data/processedData.csv')
