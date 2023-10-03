from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from models.mlp_model import MlpModel


class MlpRandomizedModel(MlpModel):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    mlp = MLPRegressor(random_state=111)

    randomized_search_cv = {
      "hidden_layer_sizes": [10, 500], 
      "activation": ["identity", "logistic", "tanh", "relu"], 
      "solver": ["lbfgs", "sgd", "adam"], 
      "alpha": [0.00005, 0.05],
      "learning_rate": ["constant", "invscaling", "adaptive"],
      "max_iter": [150, 300, 400]
    }
    
    self.model = RandomizedSearchCV(estimator=mlp,
                                    scoring='neg_root_mean_squared_error',
                                    param_distributions = randomized_search_cv,
                                    n_jobs=-1,
                                    error_score='raise',
                                    random_state=111
                                  )
  
