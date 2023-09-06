#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import argparse
from prophet import Prophet


# In[2]:


parser = argparse.ArgumentParser()

parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching')
parser.add_argument('-f', '--file', help='Path for input file. First line should contain number of lines to search in')
parser.add_argument('--predict_periods', help="How many periods to the future to predict.", type=int, default=365)
parser.add_argument('--data_file', help="File name with dataset.", type=str, default='accidents_daily.csv')

args = parser.parse_args()

periods_to_predict = int(args.predict_periods)
data_file = args.data_file


# In[ ]:





# In[3]:


dataset_df = pd.read_csv(data_file, index_col='date', parse_dates=True)

dataset_df


# In[ ]:





# In[ ]:





# In[4]:


# split the data to training set and test set

column_names = ['ds', 'y']  # ds = date, y = number of accidents in that day/month

# train set = years 2016-2021, test set = year 2022
dataset_train = dataset_df[:len(dataset_df) - periods_to_predict].reset_index()  # reserve last 12 observation as test set
dataset_train.columns = column_names
dataset_test = dataset_df[len(dataset_df) - periods_to_predict:].reset_index()
dataset_test.columns = column_names

print(dataset_train)
print(dataset_test)


# In[5]:


dataset_df.reset_index(inplace=True)
dataset_df.columns = column_names

print(dataset_df)


# In[6]:


# create a dataframe that will contain forecasts from all applied algorithms
predict_df = pd.DataFrame(dataset_df['ds'])

# add new dates to the predict_df in the length of months from the arguments
last_date = dataset_df['ds'].max()
next_dates = pd.date_range(start=last_date, periods=periods_to_predict + 1, freq='1d')
new_rows = pd.DataFrame(next_dates, columns=['ds'])
new_rows = new_rows.iloc[1:]

predict_df = pd.concat([predict_df, new_rows])

# add original values to the predict_df dataframe
predict_df['original'] = list(dataset_train['y']) + list(dataset_test['y']) + [None] * periods_to_predict
predict_df.set_index('ds', inplace=True)

predict_df


# In[7]:





# In[12]:





# In[13]:


# Prophet

prophet_model = Prophet()  # create new object for forecasting
prophet_model.fit(dataset_train)  # fit the training data
future = prophet_model.make_future_dataframe(periods=periods_to_predict + periods_to_predict, freq='1d')  # set length of forecast
forecast = prophet_model.predict(future)  # predict for the number of observations set in the previous step


# In[14]:


fig1 = prophet_model.plot(forecast)  # plot the forecast
plt.title("Prophet")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[15]:


predict_df['prophet'] = [None]*len(dataset_train) + list(forecast['yhat'].tail(periods_to_predict + periods_to_predict))  # add the forecast to predict_df

predict_df[['original', 'prophet']].plot()
plt.title("Prophet")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[ ]:





# In[ ]:





# In[16]:


# use grid search to find the most optimal hyperparameters for SVR model
param_grid = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto', 0.1, 1]
}

cv = TimeSeriesSplit(n_splits=5)

svr = SVR()

grid_search = GridSearchCV(svr, param_grid, cv=cv)
grid_search.fit(dataset_train['ds'].values.reshape(-1, 1), dataset_train['y'].values)  # fit the training data

print("Best parameters: ", grid_search.best_params_)  # print the most optimal hyperparameter values
print("Best score: ", grid_search.best_score_)


# In[17]:


# SVR
svr_model = SVR(kernel='rbf', gamma='auto', C=10, epsilon=1)  # create model with values from the GridSearchCV
svr_model.fit(dataset_train.index.values.reshape(-1, 1), dataset_train['y'].values)  # fit the training data

y_pred = svr_model.predict(dataset_test.index.values.reshape(-1, 1))

y_pred


# In[18]:


# plot the whole dataset + add predicted values for 2022
svr_predict_df = pd.DataFrame(dataset_df)
svr_predict_df['forecast_value'] = [None]*len(dataset_train) + list(y_pred)

svr_predict_df.set_index('ds', inplace=True)
svr_predict_df.columns = ["original", "predicted"]
svr_predict_df.plot()
plt.title("SVR by month")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[19]:





# In[201]:


from scalecast.Forecaster import Forecaster


# In[202]:


# create a Forecaster object that will make forecasts from multiple models
f_model = Forecaster(
    y=np.array(dataset_df.reset_index()['y']),
    current_dates=np.array(dataset_df.reset_index()['ds'])
)
f_model


# In[203]:


f_model.set_test_length(periods_to_predict)  # reserve last n observations as a test set
f_model.generate_future_dates(periods_to_predict)  # predict 12 months into the future
f_model.set_validation_length(f_model.test_length)
f_model.auto_Xvar_select()
f_model


# In[206]:


forecaster_estimators = ['arima', 'svr', 'xgboost', 'catboost', 'rnn', 'lstm']  # models that will by predicted by Forecaster
for estimator in forecaster_estimators:
    f_model.set_estimator(estimator)
    f_model.tune()  # find optimal hyperparameters for selected method
    f_model.auto_forecast()  # predict future values using hyperparameters from the previous step
    print(f'Hyperparameter values for {estimator}:')
    print(f_model.best_params)  # print the most optimal hyperparameters


# In[ ]:





# In[ ]:





# In[207]:


f_model.plot()  # plots future values
plt.title("Forecasts by multiple models")
plt.xlabel("date")
plt.ylabel("number of accidents")
f_model.plot_test_set()  # plots values from test set
plt.title("Forecasts by multiple models")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[214]:


# add forecasts from models in Forecaster to predict_df
for method in forecaster_estimators:
    result = f_model.export_fitted_vals(method)
    predict_df[method] = [None] * len(dataset_train) + list(result.tail(periods_to_predict + periods_to_predict)['FittedVals'])


# In[ ]:





# In[37]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn_genetic import GASearchCV
from sklearn_genetic import ExponentialAdapter
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution
from sklearn.metrics import accuracy_score


# In[38]:


# split datetime to day, month, and year parts for models with GA
dataset_ga = dataset_df
dataset_ga['day'] = dataset_df['ds'].dt.day
dataset_ga['month'] = dataset_df['ds'].dt.month
dataset_ga['year'] = dataset_df['ds'].dt.year
dataset_ga


# In[39]:


# split data to training and testing sets
X = dataset_ga[['day', 'month', 'year']]
y = dataset_ga['y']
X_train_ga = X[:len(X) - periods_to_predict]
X_test_ga = X[len(X) - periods_to_predict:]
y_train_ga = y[:len(y) - periods_to_predict]  # test set = observations from the last year
y_test_ga = y[len(y) - periods_to_predict:]
X_train_ga.shape, X_test_ga.shape, y_train_ga.shape, y_test_ga.shape


# In[ ]:





# In[40]:


rf = RandomForestRegressor(random_state=111)


# In[41]:


# adapters used in GA Search
mutation_adapter = ExponentialAdapter(initial_value=0.9, end_value=0.1, adaptive_rate=0.1)
crossover_adapter = ExponentialAdapter(initial_value=0.1, end_value=0.9, adaptive_rate=0.1)


# In[42]:


# grid with attributes for random forest, for genetic algorithm
rf_grid_ga = {
    'n_estimators': Integer(100, 600),
    'max_features': Categorical(['auto', 'sqrt', 'log2']),
    'max_depth': Integer(2,20),
    'criterion': Categorical(['poisson', 'squared_error', 'absolute_error', 'friedman_mse']),
    'min_samples_split': Continuous(0.1, 0.9),
    'bootstrap': Categorical([True, False])
}


# In[43]:


rf_estimator_ga = GASearchCV(estimator=rf,
                             scoring='r2',
                             population_size=100,
                             generations=12,
                             mutation_probability=mutation_adapter,
                             crossover_probability=crossover_adapter,
                             param_grid=rf_grid_ga,
                             n_jobs=-1,
                             error_score='raise'
                            )


# In[44]:


rf_estimator_ga.fit(X_train_ga, y_train_ga)  # fit the training data


# In[45]:


rf_estimator_ga.best_params_


# In[46]:


# prepare testing data with future dates in the length of argument months_to_predict
last_date = dataset_ga.ds.max()
future_dates = pd.date_range(start=last_date, periods=periods_to_predict + 1, freq='1d')
new_rows = pd.DataFrame(future_dates, columns=['ds'])
new_rows = new_rows.iloc[1:]
new_rows['day'] = new_rows['ds'].dt.day
new_rows['month'] = new_rows['ds'].dt.month
new_rows['year'] = new_rows['ds'].dt.year
X_test_ga = pd.concat([X_test_ga, new_rows])


# In[47]:


X_test_ga = X_test_ga[['day', 'month', 'year']]
X_test_ga


# In[ ]:





# In[48]:


y_predict_rf_ga = rf_estimator_ga.predict(X_test_ga)  # predict future dates


# In[49]:


y_predict_rf_ga


# In[50]:


plot_fitness_evolution(rf_estimator_ga)


# In[51]:


# prepare dataframe with forecast values (predict_df will contain all hyperparameter optimization techniques)
rf_predict_df = dataset_df[['ds']]

last_date = rf_predict_df.ds.max()
future_dates = pd.date_range(start=last_date, periods=periods_to_predict + 1, freq='1d')
new_rows = pd.DataFrame(future_dates, columns=['ds'])
new_rows = new_rows.iloc[1:]

rf_predict_df = pd.concat([rf_predict_df, new_rows])

rf_predict_df['original'] = list(y_train_ga) + list(y_test_ga) + [None] * periods_to_predict
rf_predict_df['predicted - GA'] = [None] * len(X_train_ga) + list(y_predict_rf_ga)
rf_predict_df.set_index('ds', inplace=True)


# In[52]:


# plot results with random forest and GA
rf_predict_df.plot()
plt.title("Random forest prediction")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[53]:


# create grid for randomized searach cv
rf_grid_randomized = {
    'n_estimators': [100, 600],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 20],
    'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
    'min_samples_split': [0.1, 0.9],
    'bootstrap': [True, False]
}


# In[54]:


from sklearn.model_selection import RandomizedSearchCV


# In[55]:


rf = RandomForestRegressor(random_state=111)
rf_estimator_randomized = RandomizedSearchCV(estimator=rf,
                                             scoring='r2',
                                             param_distributions = rf_grid_randomized,
                                             n_jobs=-1,
                                             error_score='raise',
                                             random_state=111
                                            )


# In[56]:


rf_estimator_randomized.fit(X_train_ga, y_train_ga)


# In[57]:


rf_estimator_randomized.best_params_


# In[58]:


y_predict_rf_randomized = rf_estimator_randomized.predict(X_test_ga)


# In[59]:


y_predict_rf_randomized


# In[60]:


rf_predict_df['predicted - randomized'] = [None] * len(X_train_ga) + list(y_predict_rf_randomized)


# In[61]:


# plot results with random forest and randomized search cv
rf_predict_df[['original', 'predicted - randomized']].plot()
plt.title("Random forest prediction")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[62]:


# grid for grid search cv
rf_grid_grid = {
    'n_estimators': [100, 200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [2, 10, 20],
    'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
    'min_samples_split': [0.1, 0.5, 0.9],
    'bootstrap': [True, False]
}


# In[63]:


from sklearn.model_selection import GridSearchCV


# In[64]:


from sklearn.model_selection import GridSearchCV
rf_estimator_grid = GridSearchCV(estimator=rf,
                                 scoring='r2',
                                 param_grid = rf_grid_grid,
                                 n_jobs=-1,
                                 error_score='raise'
                                )


# In[65]:


y_predict_rf_grid = rf_estimator_grid.fit(X_train_ga, y_train_ga)


# In[66]:


rf_estimator_grid.best_params_


# In[67]:


y_predict_rf_grid = rf_estimator_grid.predict(X_test_ga)


# In[68]:


y_predict_rf_grid


# In[69]:


rf_predict_df['predicted - grid'] = [None] * len(X_train_ga) + list(y_predict_rf_grid)


# In[70]:


# plot results with random forest and grid search cv
rf_predict_df[['original', 'predicted - grid']].plot()
plt.title("Random forest prediction")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[71]:


# r2 score with GA search cv
rf_estimator_ga.score(X_test_ga[:periods_to_predict], y_test_ga)


# In[72]:


# r2 score with randomized search cv
rf_estimator_randomized.score(X_test_ga[:periods_to_predict], y_test_ga)


# In[73]:


# r2 score with grid search cv
rf_estimator_grid.score(X_test_ga[:periods_to_predict], y_test_ga)


# In[74]:


# plot all 3 techniques for random forest on one plot
rf_predict_df.plot()
plt.title("Random Forest")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[75]:


predict_df['random forest + GA'] = list(rf_predict_df['predicted - GA'])
predict_df['random forest + randomized'] = list(rf_predict_df['predicted - randomized'])
predict_df['random forest + grid'] = list(rf_predict_df['predicted - grid'])

predict_df


# In[ ]:





# In[76]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


# In[77]:


# pipeline with data scaling and random forest
steps = [
    ('scaler', MinMaxScaler()),  # Data preprocessing step
    ('rf', RandomForestRegressor(random_state=111))  # Random Forest Regressor step
]


# In[78]:


pipeline = Pipeline(steps)


# In[79]:


# GA search cv with pipeline
rf_grid_ga = {
    'rf__n_estimators': Integer(100, 600),
    'rf__max_features': Categorical(['auto', 'sqrt', 'log2']),
    'rf__max_depth': Integer(2,20),
    'rf__criterion': Categorical(['poisson', 'squared_error', 'absolute_error', 'friedman_mse']),
    'rf__min_samples_split': Continuous(0.1, 0.9),
    'rf__bootstrap': Categorical([True, False])
}

rf_estimator_ga = GASearchCV(estimator=pipeline,
                             scoring='r2',
                             population_size=100,
                             generations=12,
                             mutation_probability=mutation_adapter,
                             crossover_probability=crossover_adapter,
                             param_grid=rf_grid_ga,
                             n_jobs=-1,
                             error_score='raise'
                            )


# In[80]:


rf_estimator_ga.fit(X_train_ga, y_train_ga)


# In[81]:


y_predict_rf_ga = rf_estimator_ga.predict(X_test_ga)


# In[82]:


y_predict_rf_ga


# In[83]:


# r2 score for random forest with GA and pipeline
rf_estimator_ga.score(X_test_ga[:periods_to_predict], y_test_ga)


# In[84]:


rf_predict_df['predicted - GA - pipe'] = [None] * len(X_train_ga) + list(y_predict_rf_ga)


# In[85]:


# plot results of random forest regressor with pipeline and GA search cv
rf_predict_df[['original', 'predicted - GA - pipe']].plot()
plt.title("Random forest prediction")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[ ]:





# In[86]:


from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing


# In[87]:


mlp = MLPRegressor(random_state=111)


# In[88]:


# adapters for MLP with GA search cv
mutation_adapter = ExponentialAdapter(initial_value=0.9, end_value=0.1, adaptive_rate=0.1)
crossover_adapter = ExponentialAdapter(initial_value=0.1, end_value=0.9, adaptive_rate=0.1)


# In[89]:


# prepare data for MLP
scaler = StandardScaler()
scaler.fit(dataset_ga[['day', 'month', 'year', 'y']].head(len(dataset_ga) - periods_to_predict))  # fit the scaler with training data

# generate new dates
new_rows['day'] = new_rows['ds'].dt.day
new_rows['month'] = new_rows['ds'].dt.month
new_rows['year'] = new_rows['ds'].dt.year
scaled_data = dataset_df[['day', 'month', 'year', 'y']]
scaled_data = pd.concat([scaled_data, new_rows[['day', 'month', 'year']]])
scaled_data = pd.DataFrame(scaler.transform(scaled_data[['day', 'month', 'year', 'y']]), columns=['day', 'month', 'year', 'y'])

X = scaled_data[['day', 'month', 'year']]
y = scaled_data['y']

# split the scaled data to train and test set
X_train_scale = X.head(len(X) - periods_to_predict - periods_to_predict)
X_test_scale = X.tail(periods_to_predict + periods_to_predict)
y_train_scale = y.head(len(X) - periods_to_predict - periods_to_predict)
y_test_scale = y.tail(periods_to_predict + periods_to_predict)
X_train_scale.shape, X_test_scale.shape, y_train_scale.shape, y_test_scale.shape


# In[ ]:





# In[90]:


# hyperparameters in MLP model with GA
mlp_grid_ga = {
    "hidden_layer_sizes": Integer(5, 200), 
    "activation": Categorical(["identity", "logistic", "tanh", "relu"]), 
    "solver": Categorical(["lbfgs", "sgd", "adam"]), 
    "alpha": Continuous(0.00005, 0.05),
    "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
    "max_iter": Integer(150, 300)
}


# In[91]:


mlp_estimator_ga = GASearchCV(estimator=mlp,
                             scoring='r2',
                             population_size=200,
                             generations=12,
                             mutation_probability=mutation_adapter,
                             crossover_probability=crossover_adapter,
                             param_grid=mlp_grid_ga,
                             n_jobs=-1,
                             error_score='raise'
                            )


# In[ ]:





# In[92]:


mlp_estimator_ga.fit(X_train_scale, y_train_scale)


# In[93]:


mlp_estimator_ga.best_params_


# In[94]:


y_predict_mlp_ga = mlp_estimator_ga.predict(X_test_scale)
y_predict_mlp_ga


# In[95]:


# r2 score for MLP with GA
mlp_estimator_ga.score(X_test_scale.head(periods_to_predict), y_test_scale.head(periods_to_predict))


# In[96]:


# create dataframe with forecast (MLP + GA)
predictions_unscaled = X_test_scale
predictions_unscaled['y'] = y_predict_mlp_ga.tolist()
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions_unscaled), columns=['day', 'month', 'year', 'y'])
predictions_unscaled


# In[97]:


mlp_predict_df = pd.concat([dataset_df[['ds']], new_rows[['ds']]], ignore_index=True)  # add results to predict_df

mlp_predict_df['original'] = list(dataset_df['y']) + [None] * periods_to_predict
mlp_predict_df['predicted - GA'] = [None] * (len(mlp_predict_df) - periods_to_predict - periods_to_predict) + list(predictions_unscaled['y'])
mlp_predict_df.set_index('ds', inplace=True)
mlp_predict_df.plot()  # plot results of MLP with GA
plt.title("MLP using GA")
plt.xlabel("date (month)")
plt.ylabel("number of accidents")


# In[98]:


plot_fitness_evolution(mlp_estimator_ga)


# In[ ]:





# In[99]:


# grid for MLP and Randomized search CV
mlp_grid_randomized = {
    "hidden_layer_sizes": [10, 500], 
    "activation": ["identity", "logistic", "tanh", "relu"], 
    "solver": ["lbfgs", "sgd", "adam"], 
    "alpha": [0.00005, 0.05],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "max_iter": [150, 300]
}


# In[100]:


from sklearn.model_selection import RandomizedSearchCV


# In[101]:


mlp_estimator_randomized = RandomizedSearchCV(estimator=mlp,
                                             scoring='neg_root_mean_squared_error',
                                             param_distributions = mlp_grid_randomized,
                                             n_jobs=-1,
                                             error_score='raise',
                                             random_state=111
                                            )


# In[102]:


mlp_estimator_randomized.fit(X_train_scale, y_train_scale)


# In[103]:


mlp_estimator_randomized.best_params_


# In[104]:


y_predict_mlp_randomized = mlp_estimator_randomized.predict(X_test_scale[['day', 'month', 'year']])
y_predict_mlp_randomized


# In[105]:


# r2 score for MLP regressor with randomized search cv
mlp_estimator_randomized.score(X_test_scale[['day', 'month', 'year']].head(periods_to_predict), y_test_scale.head(periods_to_predict))


# In[106]:


# add forecasted values to a dataframe
predictions_unscaled = X_test_scale
predictions_unscaled['y'] = y_predict_mlp_randomized.tolist()
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions_unscaled), columns=['day', 'month', 'year', 'y'])
predictions_unscaled


# In[107]:


# plot predicted values from MLP model with randomized search cv
mlp_predict_df['predicted - Randomized'] = [None] * (len(mlp_predict_df) - periods_to_predict - periods_to_predict) + list(predictions_unscaled['y'])
mlp_predict_df[['original', 'predicted - Randomized']].plot()
plt.title("MLP using Randomized search")
plt.xlabel("date (month)")
plt.ylabel("number of accidents")


# In[ ]:





# In[108]:


# grid for MLP and Grid search CV
mlp_grid = {
    "hidden_layer_sizes": [10, 250, 500], 
    "activation": ["identity", "logistic", "tanh", "relu"], 
    "solver": ["lbfgs", "sgd", "adam"], 
    "alpha": [0.00005, 0.005, 0.05],
    "learning_rate": ["constant", "invscaling", "adaptive"],
    "max_iter": [150, 300]
}


# In[109]:


mlp_estimator_grid = GridSearchCV(estimator=mlp,
                                  scoring='neg_root_mean_squared_error',
                                  param_grid = mlp_grid_randomized,
                                  n_jobs=-1,
                                  error_score='raise'
                                 )


# In[110]:


mlp_estimator_grid.fit(X_train_scale, y_train_scale)


# In[111]:


mlp_estimator_grid.best_params_


# In[112]:


y_predict_mlp_grid = mlp_estimator_grid.predict(X_test_scale[['day', 'month', 'year']])
y_predict_mlp_grid


# In[113]:


# r2 score for MLP regressor with grid search cv
mlp_estimator_grid.score(X_test_scale[['day', 'month', 'year']].head(periods_to_predict), y_test_scale.head(periods_to_predict))


# In[114]:


# add forecasted values to a dataframe
predictions_unscaled = X_test_scale
predictions_unscaled['y'] = y_predict_mlp_grid.tolist()
predictions_unscaled = pd.DataFrame(scaler.inverse_transform(predictions_unscaled), columns=['day', 'month', 'year', 'y'])
predictions_unscaled


# In[115]:


# plot predicted values from MLP model with grid search cv
mlp_predict_df['predicted - Grid'] = [None] * (len(mlp_predict_df) - periods_to_predict - periods_to_predict) + list(predictions_unscaled['y'])
mlp_predict_df[['original', 'predicted - Grid']].plot()
plt.title("MLP using Grid search")
plt.xlabel("date (month)")
plt.ylabel("number of accidents")


# In[ ]:





# In[116]:


predict_df['MLP + GA'] = list(mlp_predict_df['predicted - GA'])
predict_df['MLP + randomized'] = list(mlp_predict_df['predicted - Randomized'])
predict_df['MLP + grid'] = list(mlp_predict_df['predicted - Grid'])

predict_df


# In[ ]:





# In[259]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv


# In[260]:


def create_graph_dataset(dataframe, num_neighbors=21):
    ds_values = pd.to_datetime(dataframe['ds']).values.astype(float)
    y_values = dataframe['y'].values.astype(float)

    x = torch.tensor(y_values, dtype=torch.float32).view(-1, 1)
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    for i in range(len(ds_values)):
        # num_neighbors nearest timestamps will be edges
        start = max(0, i - num_neighbors)
        end = min(len(ds_values), i + num_neighbors + 1)
        neighbors = list(range(start, i)) + list(range(i + 1, end))
        edges = torch.tensor([[i] * len(neighbors), neighbors], dtype=torch.long)
        edge_index = torch.cat([edge_index, edges], dim=1)

    return Data(x=x, edge_index=edge_index)

# create a graph dataset from the DataFrame with train and test data
train_dataset_gnn = create_graph_dataset(dataset_train)
new_rows = predict_df.reset_index()[['ds']].tail(periods_to_predict)
new_rows['y'] = (dataset_test['y'].to_list() * ((len(new_rows) // len(dataset_test)) + 1))[:len(new_rows)]
test_dataset_gnn = create_graph_dataset(pd.concat([dataset_test, new_rows]))  # concaternanting with new_rows to add future dates

# create data loaders for training and testing
train_loader = DataLoader([train_dataset_gnn], batch_size=64)
test_loader = DataLoader([test_dataset_gnn], batch_size=64)


# In[261]:


# define the GNN model
class TimeSeriesGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TimeSeriesGNN, self).__init__()
        torch.manual_seed(111)
        self.conv1 = GCNConv(in_channels, hidden_channels)  # first layer
        self.conv2 = GCNConv(hidden_channels, out_channels)  # second layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # first GCN layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # second GCN layer
        x = self.conv2(x, edge_index)
        
        return x

# dimensions used for the GNN
input_dim = 1  # dimension of node features
hidden_dim = 64
output_dim = 1  # dimension of the predicted output

gnn_model = TimeSeriesGNN(input_dim, hidden_dim, output_dim)  # create the GNN model

loss_fn = nn.MSELoss()  # loss function used for regression

optimizer = torch.optim.Adam(gnn_model.parameters(), lr=0.01)  # optimizer used for regression

# train the model using the DataLoader object
num_epochs = 500
for epoch in range(num_epochs):
    gnn_model.train()
    for data in train_loader:
        optimizer.zero_grad()
        output = gnn_model(data)
        loss = loss_fn(output, data.x)
        loss.backward()
        optimizer.step()

# make predictions
gnn_model.eval()
for data in test_loader:
    output = gnn_model(data)
    print(output)
    predict_df['GNN'] = [None] * len(dataset_train) + output.flatten().tolist()


# In[ ]:





# In[122]:


from autots import AutoTS


# In[123]:


autots_model = AutoTS(forecast_length=periods_to_predict)
autots_model = autots_model.fit(dataset_train, date_col='ds', value_col='y', id_col=None)


# In[135]:


autots_model  # show the most optimal model and its hyperparameters


# In[136]:


prediction = autots_model.predict(forecast_length=periods_to_predict + periods_to_predict)
forecast = prediction.forecast


# In[137]:


forecast.columns = ['predicted']
forecast


# In[138]:


# add results to predict_df
predict_df['AutoTS'] = [np.nan] * len(dataset_train) + list(forecast['predicted'])

# plot the results
predict_df[['original', 'AutoTS']].plot()
plt.title('best algorithm by AutoTS')
plt.xlabel('date')
plt.ylabel('number of accidents')


# In[139]:


autots_model.results()


# In[ ]:





# In[263]:


from sklearn.metrics import mean_squared_error

target_values = predict_df.tail(periods_to_predict + periods_to_predict).head(periods_to_predict)['original'].values
rmse_results = {}

for col in predict_df.columns:
    if col != 'original':
        # Get the predicted values for the current model
        predicted_values = predict_df.tail(periods_to_predict + periods_to_predict).head(periods_to_predict)[col].values

        # Calculate the mean squared error
        mse = mean_squared_error(target_values, predicted_values)

        # Calculate the RMSE by taking the square root of the MSE
        rmse = np.sqrt(mse)

        # Store the RMSE value in the dictionary
        rmse_results[col] = rmse
        
# Convert the dictionary to a DataFrame for easier visualization
rmse_df = pd.DataFrame.from_dict(rmse_results, orient='index', columns=['RMSE'])
rmse_df.sort_values(by='RMSE')


# In[ ]:





# In[264]:


for col in predict_df.columns:
    if col != 'original':
        plt.rcParams['figure.figsize'] = [12, 7]
        predict_df[['original', col]].tail(periods_to_predict + periods_to_predict).head(periods_to_predict).plot()
        plt.title('Number of accidents - test set')
        plt.xlabel('date')
        plt.ylabel('number of accidents')
        plt.show()


# In[ ]:





# In[ ]:




