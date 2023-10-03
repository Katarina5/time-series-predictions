from abc import abstractmethod
from models.model import Model


class RandomForestModel(Model):
  @abstractmethod
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)
  
  def split_df_datetime_parts(self, df):
    df.loc[:, 'day'] = df['ds'].dt.day
    df.loc[:, 'month'] = df['ds'].dt.month
    df.loc[:, 'year'] = df['ds'].dt.year
    return df.drop('ds', axis=1)

  def preprocess_data(self):
    self.train_df = self.split_df_datetime_parts(self.train_df)
    self.predict_df = self.split_df_datetime_parts(self.predict_df)

  def fit(self):
    self.model.fit(self.train_df.drop('y', axis=1), self.train_df['y'])  # fit the training data

  def predict(self):
    return self.model.predict(self.predict_df.drop('y', axis=1))  # predict for future dates
