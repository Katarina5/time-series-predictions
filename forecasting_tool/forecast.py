import argparse
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

from models.prophet_model import ProphetModel
from models.arima_model import ArimaModel
from models.svr_model import SvrModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatboostModel
from models.rnn_model import RnnModel
from models.lstm_model import LstmModel
from models.random_forest_ga_model import RandomForestGAModel
from models.random_forest_randomized_model import RandomForestRandomizedModel
from models.random_forest_grid_model import RandomForestGridModel
from models.mlp_ga_model import MlpGAModel
from models.mlp_randomized_model import MlpRandomizedModel
from models.mlp_grid_model import MlpGridModel
from models.gnn_model import GnnModel
from models.transformer_model import TransformerModel


# create the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--predict_days', help="How many days to the future to predict.", type=int, default=14)
parser.add_argument('--data_file', help="File name with dataset located in the same folder.", type=str, default='accidents_daily.csv')
parser.add_argument('--model', help='Which model to use for predictions.', type=str, default='prophet')

args = parser.parse_args()
days_to_predict = int(args.predict_days)
data_file = args.data_file
model = args.model


# load the data
dataset_df = pd.read_csv(data_file, parse_dates=True)
dataset_df['ds'] = pd.to_datetime(dataset_df['ds'])

# test set length = 365 days or 30 % of input data, whichever is shorter
test_len = min(365, 0.3*(len(dataset_df) - days_to_predict))

# crate training and testing DataFrames
dataset_train = dataset_df.head(len(dataset_df) - test_len - days_to_predict).copy()
dataset_test = dataset_df.tail(test_len + days_to_predict).copy()

# create DataFrame with information about how long models run
timer_df = pd.DataFrame(columns=['seconds'])
timer_df.index.name = 'model'

model_dict = {
    'prophet': ProphetModel(train_df=dataset_train, predict_df=dataset_test),
    'arima': ArimaModel(train_df=dataset_train, predict_df=dataset_test),
    'svr': SvrModel(train_df=dataset_train, predict_df=dataset_test),
    'xgboost': XGBoostModel(train_df=dataset_train, predict_df=dataset_test),
    'catboost': CatboostModel(train_df=dataset_train, predict_df=dataset_test),
    'gnn': GnnModel(train_df=dataset_train, predict_df=dataset_test),
    'transformer': TransformerModel(train_df=dataset_train, predict_df=dataset_test),
    'rnn': RnnModel(train_df=dataset_train, predict_df=dataset_test),
    'lstm': LstmModel(train_df=dataset_train, predict_df=dataset_test),
    'rf_ga': RandomForestGAModel(train_df=dataset_train, predict_df=dataset_test),
    'rf_randomized': RandomForestRandomizedModel(train_df=dataset_train, predict_df=dataset_test),
    'rf_grid': RandomForestGridModel(train_df=dataset_train, predict_df=dataset_test),
    'mlp_ga': MlpGAModel(train_df=dataset_train, predict_df=dataset_test),
    'mlp_randomized': MlpRandomizedModel(train_df=dataset_train, predict_df=dataset_test),
    'mlp_grid': MlpGridModel(train_df=dataset_train, predict_df=dataset_test)
}

if model=='all':
    models = list(model_dict.keys())
else:
    models = [model]


for model_name in models:
    m = model_dict[model_name]
    m.preprocess_data()
    start = time.time()
    m.fit()
    m_forecast = m.predict()
    end = time.time()
    timer_df.loc[model_name] = end - start
    dataset_df[model_name] = [None] * len(dataset_train) + list(m_forecast)  # add the forecast to predict_df


# set date as index for easier visualisations and evaluation
dataset_df.set_index('ds', inplace=True)

# save the predictions to a CSV file
dataset_df.tail(days_to_predict)[models].to_csv('results/predictions.csv')

# evaluation
target_values = dataset_df.tail(test_len + days_to_predict).head(test_len)['y'].values
metrics_results = {}

for m in models:
  # get the predicted values for the current model
  predicted_values = dataset_df.tail(test_len + days_to_predict).head(test_len)[m].values

  # calculate the mean squared error
  mse = mean_squared_error(target_values, predicted_values)

  # calculate the RMSE by taking the square root of the MSE
  rmse = np.sqrt(mse)

  # calculate the R2 score
  r2 = r2_score(target_values, predicted_values)

  # calculate the mean absolute percentage error (MAPE)
  mape = mean_absolute_percentage_error(target_values, predicted_values)

  # store the RMSE value in the dictionary
  metrics_results[m] = {'RMSE': rmse, 'R2': r2, 'MAPE': mape}

# convert the dictionary to a DataFrame
metrics_df = pd.DataFrame.from_dict(metrics_results, orient='index', columns=['RMSE', 'R2', 'MAPE'])
metrics_df.sort_values(by='RMSE', inplace=True)

# set the index name to 'model'
metrics_df.index.name = 'model'
# save the RMSE values to a CSV file
metrics_df.to_csv('results/metrics.csv')

# plot the model results
for m in models:
    # do not plot actual values/temperature on their own
    plt.rcParams['figure.figsize'] = [12, 7]
    dataset_df.tail(test_len + days_to_predict)[['y', m]].head(test_len + min(days_to_predict, 14)).plot()
    plt.title('Number of accidents - test set - ' + m)
    plt.xlabel('date')
    plt.ylabel('number of accidents')
    # save the plot to a PNG file
    plt.savefig('results/' + m + '.png')

# save for how long each model was running to a CSV file
timer_df.to_csv('results/time.csv')

# create plot with all models only if option 'all' is selected
if model == 'all':
  dataset_df[models + ['y']].tail(test_len + days_to_predict).head(test_len + min(14, days_to_predict)).plot()
  plt.title('')
  # save the plot to a PNG file
  plt.savefig('results/all_models.png')
