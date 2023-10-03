from models.random_forest_model import RandomForestModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV


class RandomForestRandomizedModel(RandomForestModel):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    rf = RandomForestRegressor(random_state=111)

    # dictionary with attributes for random forest, for genetic algorithm
    randomized_search_cv = {
    'n_estimators': [100, 600],
    'max_features': ['sqrt', 'log2', 1.0],
    'max_depth': [2, 20],
    'criterion': ['poisson', 'squared_error', 'absolute_error', 'friedman_mse'],
    'min_samples_split': [0.1, 0.9],
    'bootstrap': [True, False]
    }
    
    self.model = RandomizedSearchCV(estimator=rf,
                                    scoring='neg_root_mean_squared_error',
                                    param_distributions = randomized_search_cv,
                                    n_jobs=-1,
                                    error_score='raise',
                                    random_state=111
                                    )
