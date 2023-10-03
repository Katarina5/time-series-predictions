from abc import abstractmethod
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from models.model import Model


class MlpModel(Model):
  @abstractmethod
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)
    self.scaler = StandardScaler()

  def split_df_datetime_parts(self, df):
    df.loc[:, 'day'] = df['ds'].dt.day
    df.loc[:, 'month'] = df['ds'].dt.month
    df.loc[:, 'year'] = df['ds'].dt.year
    return df.drop('ds', axis=1)
  
  def preprocess_data(self):
    self.train_df = self.split_df_datetime_parts(self.train_df)
    self.predict_df = self.split_df_datetime_parts(self.predict_df)

    self.scaler.fit(self.train_df)  # fit the scaler with training data

    self.train_df = pd.DataFrame(self.scaler.transform(self.train_df), columns = self.train_df.columns)
    self.predict_df = pd.DataFrame(self.scaler.transform(self.predict_df), columns = self.predict_df.columns)

  def fit(self):
    self.model.fit(self.train_df.drop('y', axis=1), self.train_df['y'])  # fit the training data
    
  def predict(self):
    res = self.predict_df
    res['y'] = self.model.predict(self.predict_df.drop('y', axis=1))  # predict for future dates
    return pd.DataFrame(self.scaler.inverse_transform(res), columns = self.predict_df.columns)['y']
