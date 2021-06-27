import pandas as pd
from src import config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

df1 = pd.read_csv(config.data_processed)
corrmatrix=df1.corr()

x=correlateddata.drop(labels=["MEDV"], axis=1)
y=correlateddata["MEDV"]



X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=0) # splitting the dataset
X_train.shape,X_test.shape

model=LinearRegression()
model.fit(X_train, y_train) # tranning our model


y_predict=model.predict(X_test) # predicting the model
df=pd.DataFrame(data= [y_predict,y_test]) # looking the value of test and predicted value side by side
df.T

score=r2_score(y_test,y_predict)
mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
print('R2 score', score)
print('mae:',mae)
print('mse:',mse)

total_feature_trained = []
total_feature_name = []
selected_correlation_value = []
r2_scores = []
mae_value = []
mse_value = []


def performance_metrics(features, th, y_true, y_pred):
    """Function for calculating performance matrix for different features selected"""
    scores = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    total_feature_trained.append(len(features) - 1)
    total_feature_name.append(str(features))
    selected_correlation_value.append(th)
    r2_scores.append(score)
    mae_value.append(mae)
    mse_value.append(mse)

    dataframe_metrics = pd.DataFrame(
        data=[total_feature_name, total_feature_trained, selected_correlation_value, r2_scores,
              mae_value, mse_value],
        index=['Feature Name', 'Total Feature', 'Corr value', 'r2 score', 'MAE', 'MSE'])
    return dataframe_metrics.T


performance_metrics(correlateddata.columns.values, threshold, y_test, y_predict)

# regression plot of featues selected

# rows=2
# cols=2
# fig,ax=plt.subplots(nrows=rows,ncols=cols,figsize=(10,5))
# col=correlateddata.columns
# index=0
#
# for i in range(rows):
#     for j in range(cols):
#         sns.regplot(x=correlateddata[col[index]],y=correlateddata['MEDV'], ax=ax[i][j])
#         index =index+1
# fig.tight_layout

threshold=0.60
corr_values=correlatedfeature(corrmatrix['MEDV'], threshold)
corr_values

correlateddata=df1[corr_values.index]
correlateddata.head()


def get_ypredict(corrdata):
    """ Function for trainning, fitting and predicting model parametes"""
    X=corrdata.drop(labels=['MEDV'], axis=1)
    y=corrdata["MEDV"]
    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2, random_state=0)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_predict=model.predict(X_test)
    return y_predict

y_predict=get_ypredict(correlateddata)
performance_metrics(correlateddata.columns.values, threshold,y_test,y_predict)

# selecting different features with correlated value above 0.7 absolute
threshold=0.70
corr_values=correlatedfeature(corrmatrix['MEDV'], threshold)
corr_values

correlateddata=df1[corr_values.index]
correlateddata.head()


y_predict=get_ypredict(correlateddata)
performance_metrics(correlateddata.columns.values, threshold,y_test,y_predict)


# selecting different features with correlated value above 0.4 absolute
threshold=0.40
corr_values=correlatedfeature(corrmatrix['MEDV'], threshold)
corr_values

correlateddata=df1[corr_values.index]
correlateddata.head()

y_predict=get_ypredict(correlateddata)
performance_metrics(correlateddata.columns.values, threshold,y_test,y_predict)

