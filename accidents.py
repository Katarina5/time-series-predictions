#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import argparse


# In[2]:


parser = argparse.ArgumentParser()

parser.add_argument('strings', metavar='STRING', nargs='*', help='String for searching')
parser.add_argument('-f', '--file', help='Path for input file. First line should contain number of lines to search in')
parser.add_argument('--predict_days', help="How many days to the future from 1.1.2023 to predict.", type=int, default=365)
parser.add_argument('--predict_months', help="How many months to the future from 1.1.2023 to predict.", type=int, default=12)

args = parser.parse_args()

days_to_predict = int(args.predict_days)
months_to_predict = int(args.predict_months)
print(days_to_predict)
print(months_to_predict)


# In[ ]:





# In[3]:


column_names = ["identification", "ground_communication_type", "ground_communication_number", "day_month_year", 
                "weekday", "time", "accident_type", "type_of_collision_of_vehicles", "type_of_obstacle", 
                "life_consequences", "culprits_of_accident", "alcohol_in_culprit", "main_cause_of_accident", 
                "number_of_persons_died", "number_of_seriously_injured", "number_of_lightly_injured", 
                "total_material_damage", "type_of_road_surface", "condition_of_road_surface", 
                "condition_of_communication", "wind_condition", "visibility", "p20", "p21", "p22", "p23", "p24", 
                "p27", "p28", "p34", "p35", "p39", "p44", "p45a", "p47", "p48a", "p49", "p50a", "p50b", "p51", 
                "p52", "p53", "p55a", "p57", "p58", "a", "b", "d", "e", "f", "g", "h", "i", "j", "k", "l", "n", 
                "o", "p", "q", "r", "s", "t", "p5a"]
df_list = []

filepaths = ['./data-nehody/data_GIS_12-' + str(y) for y in range(2016, 2023)]

for path in filepaths:
    csv_files = glob.glob(path + "/*.csv")
    dfs_in_file = [pd.read_csv(file, sep=";", encoding='ANSI', header=None, low_memory=False) for file in csv_files]
    df_list.append(pd.concat(dfs_in_file, axis=0, ignore_index=True))

accidents_df = pd.concat(df_list, axis=0, ignore_index=True)
accidents_df.columns = column_names
accidents_df.loc[accidents_df.p47=='XX', 'p47'] = np.nan
accidents_df.loc[accidents_df.time>2400, 'time'] = 0
accidents_df.loc[accidents_df.time%100>=60, 'time'] = 0
accidents_df['time'] = accidents_df['time'].map(str).str.pad(4, fillchar='0').map(lambda t: t[:2] + ':' + t[2:])
accidents_df['date'] = pd.to_datetime(accidents_df['day_month_year'] + ' ' + accidents_df['time'])


# In[4]:


accidents_df.head(5)


# In[5]:


# during the weekend, i.e. saturday (6) and sunday (0), less accidents. 
plt.hist(accidents_df['weekday'], bins=7, rwidth=0.8)
plt.title("Number of accidents by weekday")
plt.xlabel("weekday (0 = Sunday)")
plt.ylabel("number of accidents")


# In[6]:


# count number of accidents for each day/month/year
daily_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.date)['date'].count())
monthly_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count())
yearly_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.to_period('Y'))['date'].count())


# In[7]:


# seasonal decomposition
seasonal_decompose(daily_count, period=365).plot()


# In[8]:


# plot number of accidents on each day
plt.rcParams['figure.figsize'] = [20, 5]
daily_count.plot()
plt.title("Accidents by day")
plt.xlabel("day")
plt.ylabel("number of accidents")


# In[9]:


# plot number of accidents in each month
plt.rcParams['figure.figsize'] = [8, 4]
monthly_count.plot()
plt.title("Accidents by month")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[10]:


# plot number of accidents in each year
yearly_count.plot()
plt.title("Accidents by year")
plt.xlabel("year")
plt.ylabel("number of accidents")


# In[11]:


# split the data to training set and test set
# train set = years 2016-2021, test set = year 2022
monthly_count_train = monthly_count[:len(monthly_count)-12]
monthly_count_test = monthly_count[len(monthly_count)-12:]

print(monthly_count_train)
print(monthly_count_test)


# In[12]:


# seasonal arima with automatic parameter calculation
# seasonality period = 12 months (one year)
auto_arima = pm.auto_arima(monthly_count_train, seasonal=True, m=12)
auto_arima


# In[13]:


# SARIMA
forecast_test_auto = auto_arima.predict(n_periods=len(monthly_count_test) + months_to_predict)  # predict one year from the test set and 12 months in the future
predict_df = pd.DataFrame(monthly_count)

# add future dates to predict_df
last_date = predict_df.index.max()
future_dates = pd.date_range(start=last_date.to_timestamp(), periods=months_to_predict + 1, freq='M')
new_rows = pd.DataFrame(index=future_dates, columns=predict_df.columns)
new_rows = new_rows.iloc[1:]
predict_df = pd.concat([predict_df, new_rows])

predict_df['forecast_value'] = [None]*len(monthly_count_train) + list(forecast_test_auto)

predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("seasonal ARIMA by month")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[ ]:





# In[14]:


# prepare data for Prophet
# Prophet needs a dataframe with 2 columns: ds (date) and y (number of accidents)
daily_count = accidents_df.groupby(accidents_df['date'].dt.date)['date'].count().to_frame()
daily_count.columns = ['y']
daily_count = daily_count.reset_index()
daily_count.columns = ['ds', 'y']

monthly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count().to_frame()
monthly_count.columns = ['y']
monthly_count = monthly_count.reset_index()
monthly_count.columns = ['ds', 'y']
monthly_count['ds'] = monthly_count['ds'].dt.to_timestamp()

yearly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('Y'))['date'].count().to_frame()

# split the prepared data to training and test set
daily_count_train = daily_count[:len(daily_count)-365]
daily_count_test = daily_count[len(daily_count)-365:]

monthly_count_train = monthly_count[:len(monthly_count)-12]
monthly_count_test = monthly_count[len(monthly_count)-12:]

print(daily_count)


# In[15]:


# Prophet - daily

m = Prophet()
m.fit(daily_count_train)
future = m.make_future_dataframe(periods=365 + days_to_predict)


# In[16]:


forecast = m.predict(future)


# In[17]:


forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(24)

fig1 = m.plot(forecast)
plt.title("Prophet by day")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[18]:


fig2 = m.plot_components(forecast)


# In[19]:


# plot the whole dataset + add predicted values for 2022
daily_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.date)['date'].count())
predict_df = pd.DataFrame(daily_count)

# add future dates to predict_df
last_date = predict_df.index.max()
future_dates = pd.date_range(start=last_date, periods=1 + days_to_predict, freq='d')
new_rows = pd.DataFrame(index=future_dates, columns=predict_df.columns)
new_rows = new_rows.iloc[1:]
predict_df = pd.concat([predict_df, new_rows])

predict_df['forecast_value'] = [None]*len(daily_count_train) + list(forecast['yhat'].tail(365 + days_to_predict))

plt.rcParams['figure.figsize'] = [20, 5]
predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("Prophet by day")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[20]:


# Prophet - monthly

m = Prophet()
m.fit(monthly_count_train)
future = m.make_future_dataframe(periods=12 + months_to_predict, freq='MS')
forecast = m.predict(future)


# In[21]:


fig1 = m.plot(forecast)
plt.title("Prophet by month")


# In[22]:


# plot the whole dataset + add predicted values for 2022
monthly_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count())
predict_df = pd.DataFrame(monthly_count)

# add future dates to predict_df
last_date = predict_df.index.max()
future_dates = pd.date_range(start=last_date.to_timestamp(), periods=months_to_predict + 1, freq='M')
new_rows = pd.DataFrame(index=future_dates, columns=predict_df.columns)
new_rows = new_rows.iloc[1:]
predict_df = pd.concat([predict_df, new_rows])

predict_df['forecast_value'] = [None]*len(monthly_count_train) + list(forecast['yhat'].tail(months_to_predict + 12))

predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("Prophet by month")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[ ]:





# In[23]:


# prepare dataset for SVR
daily_count = accidents_df.groupby(accidents_df['date'].dt.date)['date'].count().to_frame()
daily_count.columns = ['y']
daily_count.index = pd.to_datetime(daily_count.index)

monthly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count().to_frame()
monthly_count.columns = ['y']
monthly_count = monthly_count.reset_index()
monthly_count.columns = ['ds', 'y']
monthly_count['ds'] = monthly_count['ds'].dt.to_timestamp()
monthly_count = monthly_count.set_index('ds')

yearly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('Y'))['date'].count().to_frame()

# split data to training set and test set
daily_count_train = daily_count[:len(daily_count)-365]
daily_count_test = daily_count[len(daily_count)-365:]

monthly_count_train = monthly_count[:len(monthly_count)-12]
monthly_count_test = monthly_count[len(monthly_count)-12:]

print(daily_count_train)


# In[24]:


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
grid_search.fit(monthly_count_train.index.values.reshape(-1, 1), monthly_count_train['y'].values)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[25]:


# SVR - monthly
model = SVR(kernel='sigmoid',gamma='scale', C=0.1, epsilon=0.01)
model.fit(monthly_count_train.index.values.reshape(-1, 1), monthly_count_train['y'].values)

y_pred = model.predict(monthly_count_test.index.values.reshape(-1, 1))

y_pred


# In[26]:


# plot the whole dataset + add predicted values for 2022
monthly_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count())
predict_df = pd.DataFrame(monthly_count)
predict_df['forecast_value'] = [None]*len(monthly_count_train) + list(y_pred)

predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("SVR by month")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[27]:


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
grid_search.fit(daily_count_train.index.values.reshape(-1, 1), daily_count_train['y'].values)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[28]:


# SVR - daily
model = SVR(kernel='rbf',gamma='auto', C=10, epsilon=1)
model.fit(daily_count_train.index.values.reshape(-1, 1), daily_count_train['y'].values)

y_pred = model.predict(daily_count_test.index.values.reshape(-1, 1))

y_pred


# In[29]:


# plot the whole dataset + add predicted values for 2022
daily_count = pd.DataFrame(accidents_df.groupby(accidents_df['date'].dt.date)['date'].count())
predict_df = pd.DataFrame(daily_count)
predict_df['forecast_value'] = [None]*len(daily_count_train) + list(y_pred)

plt.rcParams['figure.figsize'] = [20, 5]
predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("SVR by day")
plt.xlabel("date")
plt.ylabel("number of accidents")


# In[29]:


# scaling the data for SVR
window_size = 3

# Compute rolling mean and standard deviation
rolling_mean = monthly_count_train['y'].rolling(window=window_size).mean()
rolling_std = monthly_count_train['y'].rolling(window=window_size).std()

# Apply scaling transformation to each time step
scaled_data = (monthly_count_train['y'] - rolling_mean) / rolling_std

scaled_data

monthly_count_train['y'] = scaled_data


# In[30]:


monthly_count_train = monthly_count_train.iloc[2:]
monthly_count_train


# In[31]:


#find optimal hyperparameters susing grid search for scaled data
param_grid = {
    'kernel': ['rbf', 'sigmoid'],
    'C': [0.1, 1, 10],
    'epsilon': [0.01, 0.1, 1],
    'gamma': ['scale', 'auto', 0.1, 1]
}

cv = TimeSeriesSplit(n_splits=5)

svr = SVR()

grid_search = GridSearchCV(svr, param_grid, cv=cv)
grid_search.fit(monthly_count_train.index.values.reshape(-1, 1), monthly_count_train['y'].values)

print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)


# In[32]:


# SVR - monhtly with scaled data
model = SVR(kernel='rbf',gamma='auto', C=1, epsilon=1)
model.fit(monthly_count_train.index.values.reshape(-1, 1), monthly_count_train['y'].values)

y_pred = model.predict(monthly_count_test.index.values.reshape(-1, 1))

y_pred


# In[ ]:





# In[13]:





# In[30]:


from scalecast.Forecaster import Forecaster


# In[31]:


monthly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count().to_frame()
monthly_count.columns = ['y']
monthly_count = monthly_count.reset_index()
monthly_count.columns = ['ds', 'y']
monthly_count['ds'] = monthly_count['ds'].dt.to_timestamp()
monthly_count = monthly_count.set_index('ds')
monthly_count[["y"]] = monthly_count[["y"]].values.astype('float32')


# In[32]:


daily_count = accidents_df.groupby(accidents_df['date'].dt.date)['date'].count().to_frame()
daily_count.columns = ['y']
daily_count.index = pd.to_datetime(daily_count.index)
daily_count[["y"]] = daily_count[["y"]].values.astype('float32')
daily_count = daily_count.reset_index()
daily_count.columns = ['ds', 'y']


# In[33]:


f_daily = Forecaster(
    y=np.array(daily_count['y']),
    current_dates=np.array(daily_count['ds'])
)
f_daily


# In[34]:


f_daily.set_test_length(365)  # reserve last 365 observations (a year) as a test set
f_daily.generate_future_dates(days_to_predict)  # predict specified number of days into the future
f_daily.set_validation_length(f_daily.test_length)
f_daily.auto_Xvar_select()
f_daily


# In[35]:


for method in ['arima', 'svr', 'mlp', 'lstm']:
    f_daily.set_estimator(method)
    f_daily.tune()  # find optimal hyperparameters for selected method
    f_daily.auto_forecast()  # predict future values using hyperparameters from the previous step
    print(f'Hyperparameter values for {method}:')
    print(f_daily.best_params)


# In[36]:


f_daily.plot()  # plots future values
f_daily.plot_test_set()  # plots values from test set


# In[38]:


# models orderet from best performing to worst performing
f_daily.order_fcsts()


# In[ ]:





# In[39]:


f_monthly = Forecaster(
    y=np.array(monthly_count.reset_index()['y']),
    current_dates=np.array(monthly_count.reset_index()['ds'])
)
f_monthly


# In[40]:


f_monthly.set_test_length(12)  # reserve last 12 observations (a year) as a test set
f_monthly.generate_future_dates(months_to_predict)  # predict 12 months into the future
f_monthly.set_validation_length(f_monthly.test_length)
f_monthly.auto_Xvar_select()
f_monthly


# In[41]:


for method in ['svr', 'mlp', 'lstm']:
    f_monthly.set_estimator(method)
    f_monthly.tune()  # find optimal hyperparameters for selected method
    f_monthly.auto_forecast()  # predict future values using hyperparameters from the previous step
    print(f'Hyperparameter values for {method}:')
    print(f_monthly.best_params)


# In[42]:


f_monthly.plot()  # plots future values
f_monthly.plot_test_set()  # plots values from test set


# In[4]:


from autots import AutoTS


# In[6]:


# prepare data for AutoTS
daily_count = accidents_df.groupby(accidents_df['date'].dt.date)['date'].count().to_frame()
daily_count.columns = ['y']
daily_count = daily_count.reset_index()
daily_count.columns = ['ds', 'y']

monthly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('M'))['date'].count().to_frame()
monthly_count.columns = ['y']
monthly_count = monthly_count.reset_index()
monthly_count.columns = ['ds', 'y']
monthly_count['ds'] = monthly_count['ds'].dt.to_timestamp()

yearly_count = accidents_df.groupby(accidents_df['date'].dt.to_period('Y'))['date'].count().to_frame()

# split the prepared data to training and test set
daily_count_train = daily_count[:len(daily_count)-365]
daily_count_test = daily_count[len(daily_count)-365:]

monthly_count_train = monthly_count[:len(monthly_count)-12]
monthly_count_test = monthly_count[len(monthly_count)-12:]

print(monthly_count)


# In[45]:


model = AutoTS(forecast_length=months_to_predict)
model = model.fit(monthly_count_train, date_col='ds', value_col='y', id_col=None)


# In[46]:


model


# In[47]:


prediction = model.predict(forecast_length=months_to_predict + 12)
forecast = prediction.forecast


# In[48]:


forecast.columns = ['predicted']
forecast


# In[49]:


# visualisation
predict_df = pd.DataFrame(monthly_count).set_index('ds')
predict_df['predicted'] = [np.nan] * len(monthly_count_train) + list(forecast.head(12)['predicted'])

new_rows = forecast.tail(len(forecast) - 12)
new_rows.insert(0, 'y', np.nan)

predict_df = pd.concat([predict_df, new_rows])

predict_df.columns = ["original", "predicted"]
predict_df.plot(marker='x')
plt.title("best algorithm selected with GP")
plt.xlabel("month")
plt.ylabel("number of accidents")


# In[7]:


model_daily = AutoTS(forecast_length=30)
model_daily = model_daily.fit(daily_count_train, date_col='ds', value_col='y', id_col=None)


# In[8]:


model_daily


# In[9]:


prediction = model_daily.predict(forecast_length=365 + days_to_predict)
forecast = prediction.forecast
forecast.columns = ['predicted']
forecast


# In[10]:


# visualisation
predict_daily_df = pd.DataFrame(daily_count).set_index('ds')
predict_daily_df['predicted'] = [np.nan] * len(daily_count_train) + list(forecast.head(365)['predicted'])

new_rows = forecast.tail(len(forecast) - 365)
new_rows.insert(0, 'y', np.nan)

predict_daily_df = pd.concat([predict_daily_df, new_rows])

predict_daily_df.columns = ["original", "predicted"]
predict_daily_df.plot(marker='x')
plt.title("best algorithm selected with GP")
plt.xlabel("day")
plt.ylabel("number of accidents")


# In[ ]:





# In[ ]:




