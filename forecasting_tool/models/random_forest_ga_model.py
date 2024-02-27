from models.random_forest_model import RandomForestModel
from sklearn.ensemble import RandomForestRegressor
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic import ExponentialAdapter, GASearchCV
from sklearn_genetic.plots import plot_fitness_evolution
import matplotlib.pyplot as plt


class RandomForestGAModel(RandomForestModel):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    rf = RandomForestRegressor(random_state=111)

    # dictionary with attributes for random forest, for genetic algorithm
    ga_search_cv = {
        'n_estimators': Integer(100, 600),
        'max_features': Categorical(['sqrt', 'log2', 1.0]),
        'max_depth': Integer(2,20),
        'criterion': Categorical(['poisson', 'squared_error', 'absolute_error', 'friedman_mse']),
        'min_samples_split': Continuous(0.1, 0.9),
        'bootstrap': Categorical([True, False])
    }

    # adapters used in GA Search
    mutation_adapter = ExponentialAdapter(initial_value=0.9, end_value=0.1, adaptive_rate=0.1)
    crossover_adapter = ExponentialAdapter(initial_value=0.1, end_value=0.9, adaptive_rate=0.1)

    self.model = GASearchCV(estimator=rf,
                            scoring='neg_root_mean_squared_error',
                            population_size=100,
                            generations=12,
                            mutation_probability=mutation_adapter,
                            crossover_probability=crossover_adapter,
                            param_grid=ga_search_cv,
                            n_jobs=-1,
                            error_score='raise'
                            )

  def fit(self):
    self.model.fit(self.train_df.drop('y', axis=1), self.train_df['y'])  # fit the training data

    # plot the evolution of the fitness function
    plot_fitness_evolution(self.model)
    plt.savefig('results/rf_ga_fitness_evolution.png', bbox_inches='tight')
    plt.clf()
