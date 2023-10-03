from autots import AutoTS, create_regressor
from models.model import Model
import pandas as pd


class AutoTSModel(Model):
    def __init__(self, train_df, predict_df) -> None:
        super().__init__(train_df, predict_df)
        self.model = AutoTS(forecast_length=self.predict_df['y'].isna().sum(),
                            model_list='fast',
                            max_generations=12)
    
    def preprocess_data(self):
        return super()

    def fit(self):
      future_regressor_train, _ = create_regressor(
          pd.concat([self.train_df.set_index('ds')[['temp']], self.predict_df.set_index('ds')[['temp']]]),
          forecast_length=len(self.predict_df)
        )
      self.model = self.model.fit(self.train_df, date_col='ds', value_col='y', id_col=None, future_regressor=future_regressor_train)
    
    def predict(self):
        _, future_regressor_forecast = create_regressor(
          pd.concat([self.train_df.set_index('ds')[['temp']], self.predict_df.set_index('ds')[['temp']]]),
          forecast_length=len(self.predict_df)
        )
        prediction = self.model.predict(forecast_length=len(self.predict_df), future_regressor=future_regressor_forecast)
        forecast = prediction.forecast
        forecast.columns = ['predicted']
        return forecast['predicted']

