from models.model import Model
from prophet import Prophet


class ProphetModel(Model):
    def __init__(self, train_df, predict_df) -> None:
        super().__init__(train_df, predict_df)
        self.model = Prophet()
        regressor_names = train_df.columns.tolist()
        regressor_names = [r for r in regressor_names if r not in ['ds', 'y']]
        for r in regressor_names:
          self.model.add_regressor(r, standardize=False)  # include other regressors in the model
    
    def preprocess_data(self):
        self.predict_df = self.predict_df.drop('y', axis=1)

    def fit(self):
        self.model.fit(self.train_df)  # fit the training data
    
    def predict(self):
        return self.model.predict(self.predict_df)['yhat']
    