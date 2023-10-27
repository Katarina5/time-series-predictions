from abc import ABC, abstractmethod


class Model(ABC):
    @abstractmethod
    def __init__(self, train_df, predict_df) -> None:
        super().__init__()
        self.train_df = train_df.copy()  # training data
        self.predict_df = predict_df.copy()  # testing data + future timesteps

    @abstractmethod
    def preprocess_data(self):
      pass
  
    @abstractmethod
    def fit(self):
        pass
  
    @abstractmethod
    def predict(self):
        pass