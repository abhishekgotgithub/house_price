from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import numpy as np

df= pd.read_csv(config.data_processed)
X = df.iloc[:,:-1] # selecting independent features
y = df.iloc[:,-1] #selecting dependent feature

x_training_set, x_test_set, y_training_set, y_test_set = train_test_split(X,y,test_size=0.20,random_state=42,shuffle=True)

dt = DecisionTreeRegressor(random_state = 42)
from pprint import pprint
# Look at parameters used by our current forest
print('Parameters currently in use:\n')
pprint(dt.get_params())

from sklearn.model_selection import RandomizedSearchCV

# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 500, num = 50)]
# Maximun leaf node
max_leaf_nodes=[int(x) for x in np.linspace(2, 500, num = 50)]
# ccp_alpha
ccp_alpha= [float(x) for x in np.linspace(0.0, 5.0, num = 5)]
#min impurity deccrease
min_impurity_decrease= [float(x) for x in np.linspace(0.0, 5.0, num = 5)]
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(2, 500, num = 50)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(2, 500, num = 50)]
# criterion
criterion = ["mse", "mae","friedman_mse"]
# Create the random grid
random_grid = {
 'criterion': criterion,
 'max_depth': max_depth,
 'ccp_alpha': ccp_alpha,
 'min_impurity_decrease' : min_impurity_decrease,
 'max_features': max_features,
 'max_leaf_nodes': max_leaf_nodes,
 'min_samples_leaf': min_samples_leaf,
 'min_samples_split': min_samples_split,
 }

# Use the random grid to search for best hyperparameters
# Random search of parameters, using 5 fold cross validation,
# search across different combinations, and use all available cores
dt_random = RandomizedSearchCV(estimator = dt, param_distributions = random_grid, n_iter = 5000, scoring=None, cv = 5, verbose=2, random_state=42, n_jobs = -1)
# Fit the random search model
dt_random.fit(x_training_set, y_training_set)


def evaluate(model, x_test, y_test):
    """Function for evaluating the performance for diff hyperparameters"""
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = dt
base_model.fit(x_training_set, y_training_set)
base_accuracy = evaluate(base_model, x_test_set, y_test_set)

best_random = DecisionTreeRegressor(min_samples_split=32,
                                    min_samples_leaf=12,
                                    min_impurity_decrease=0.0,
                                    max_leaf_nodes=42,
                                    max_features='auto',
                                    max_depth=134,
                                    ccp_alpha=0.0,
                                    criterion='friedman_mse')
best_random.fit(x_training_set, y_training_set)
random_accuracy = evaluate(best_random, x_test_set, y_test_set)

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))


def evaluate(model, x_test, y_test):
    """Function for evaluating the performance for diff hyperparameters"""
    predictions = model.predict(x_test)
    errors = abs(predictions - y_test)
    mape = 100 * np.mean(errors / y_test)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy


base_model = dt
base_model.fit(x_training_set, y_training_set)
base_accuracy = evaluate(base_model, x_test_set, y_test_set)

best_random = DecisionTreeRegressor(min_samples_split=32,
                                    min_samples_leaf=12,
                                    min_impurity_decrease=0.0,
                                    max_leaf_nodes=42,
                                    max_features='auto',
                                    max_depth=134,
                                    ccp_alpha=0.0,
                                    criterion='friedman_mse')
best_random.fit(x_training_set, y_training_set)
random_accuracy = evaluate(best_random, x_test_set, y_test_set)

print('Improvement of {:0.2f}%.'.format(100 * (random_accuracy - base_accuracy) / base_accuracy))

# Number of features to consider at every split
max_features = ['auto']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(130, 140, num = 11)]
# Maximun leaf node
max_leaf_nodes=[int(x) for x in np.linspace(38, 48, num = 11)]
# ccp_alpha
ccp_alpha= [float(x) for x in np.linspace(0.0, 5.0, num = 5)]
#min impurity deccrease
min_impurity_decrease= [0.0]
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(25, 35, num = 11)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [int(x) for x in np.linspace(7, 17, num = 11)]
# criterion
criterion = ["friedman_mse"]


from sklearn.model_selection import GridSearchCV
#from sklearn.model_selection import RandomizedSearchCV
# Create the parameter grid based on the results of random search
random_grid_cv = {'criterion': criterion,
 'max_depth': max_depth,
 'ccp_alpha': ccp_alpha,
 'min_impurity_decrease' : min_impurity_decrease,
 'max_features': max_features,
 'max_leaf_nodes': max_leaf_nodes,
 'min_samples_leaf': min_samples_leaf,
 'min_samples_split': min_samples_split,
  }
# Create a based model
dt = DecisionTreeRegressor(random_state=42)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = dt, param_grid = random_grid_cv,
                          cv = 3, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
grid_search.fit(x_training_set, y_training_set)

base_model = dt
base_model.fit(x_training_set, y_training_set)
base_accuracy = evaluate(base_model,x_test_set,y_test_set)

best_grid = grid_search.best_estimator_
grid_accuracy = evaluate(best_grid, x_test_set,y_test_set)

print('Improvement of {:0.2f}%.'.format( 100 * (grid_accuracy - base_accuracy) / base_accuracy))

