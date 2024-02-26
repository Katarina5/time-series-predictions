from sklearn.neural_network import MLPRegressor
from sklearn_genetic import ExponentialAdapter, GASearchCV
from sklearn_genetic.space import Continuous, Categorical, Integer
from sklearn_genetic.plots import plot_fitness_evolution
from models.mlp_model import MlpModel
import matplotlib.pyplot as plt


class MlpGAModel(MlpModel):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    mlp = MLPRegressor(random_state=111)

    # adapters for GA search cv
    mutation_adapter = ExponentialAdapter(initial_value=0.9, end_value=0.1, adaptive_rate=0.1)
    crossover_adapter = ExponentialAdapter(initial_value=0.1, end_value=0.9, adaptive_rate=0.1)

    ga_search_cv = {
      "hidden_layer_sizes": Integer(5, 200), 
      "activation": Categorical(["identity", "logistic", "tanh", "relu"]), 
      "solver": Categorical(["lbfgs", "sgd", "adam"]), 
      "alpha": Continuous(0.00005, 0.05),
      "learning_rate": Categorical(["constant", "invscaling", "adaptive"]),
      "max_iter": Integer(150, 400)
      }
    
    self.model = GASearchCV(estimator=mlp,
                             scoring='neg_root_mean_squared_error',
                             population_size=200,
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
    plt.rcParams['figure.figsize'] = [12, 7]
    plot_fitness_evolution(self.model)
    plt.savefig('results/mlp_ga_fitness_evolution.png', bbox_inches='tight')
    plt.clf()
