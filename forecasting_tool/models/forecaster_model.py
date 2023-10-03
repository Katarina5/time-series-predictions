from abc import abstractmethod
from models.model import Model
from scalecast.Forecaster import Forecaster
import numpy as np
import pandas as pd


class ForecasterModel(Model):
    @abstractmethod
    def __init__(self, train_df, predict_df, model_name) -> None:
        super().__init__(train_df, predict_df)
        self.model = Forecaster(
            y=np.array(train_df['y']),
            current_dates=np.array(train_df['ds']),
            metrics=['rmse']
            )
        self.model_name = model_name
        self.model.set_grids_file('models.Grids')
        self.model.set_estimator(model_name)
        self.model.set_test_length(0)  # testing will be done manually in the end
        self.model.generate_future_dates(len(self.predict_df))  # predict for test set as well as for the future
        self.model.set_validation_length(min(365, 0.3 * len(self.train_df)))

        regressor_names = train_df.columns.tolist()
        regressor_names = [r for r in regressor_names if r not in ['ds', 'y']]
        for r in regressor_names:
            self.model.add_series(pd.concat([self.train_df[r], self.predict_df[r]]), called=r)  # add regressor with temperatures


    def preprocess_data(self):
      pass  # no need to preprocess for Forecaster
  
    def fit(self):
        self.model.auto_Xvar_select()  # automatic selection of the most optimal regressors
        self.model.tune()  # find optimal hyperparameters for the current model
  
    def predict(self):
        self.model.auto_forecast()  # automatic forecasting using the most optimal regressors and hyperparameters
        return self.model.export_fitted_vals(self.model_name)['FittedVals'].tail(len(self.predict_df))
