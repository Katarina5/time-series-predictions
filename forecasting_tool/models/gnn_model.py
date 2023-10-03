from models.model import Model
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader


class GnnModel(Model):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    input_dim = len(train_df.columns) - 1  # number  of dimensions is all columns except the date
    hidden_dim = 64
    output_dim = len(train_df.columns) - 1

    self.model = TimeSeriesGNN(input_dim, hidden_dim, output_dim)  # create the GNN model

  def create_graph_dataset(self, df, num_neighbors=21):
    ds_values = pd.to_datetime(df['ds']).values.astype(float)
    regressor_names = df.columns.tolist()
    regressor_names = [r for r in regressor_names if r not in ['ds']]
    regressors = []
    for r in regressor_names:
      regressors.append(df[r].values.astype(float))

    x = torch.tensor(np.column_stack(tuple(regressors)), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)

    for i in range(len(ds_values)):
        # num_neighbors nearest timestamps will be edges
        start = max(0, i - num_neighbors)
        end = min(len(ds_values), i + num_neighbors + 1)
        neighbors = list(range(start, i)) + list(range(i + 1, end))
        edges = torch.tensor([[i] * len(neighbors), neighbors], dtype=torch.long)
        edge_index = torch.cat([edge_index, edges], dim=1)

    return Data(x=x, edge_index=edge_index)

  def preprocess_data(self):
    # create graph datasets from DataFrames
    self.train_df = self.create_graph_dataset(self.train_df)
    test_vals = self.predict_df['y'].dropna().values
    self.predict_df['y'] = np.tile(test_vals, (len(self.predict_df) // len(test_vals)) + 1)[:len(self.predict_df)]  # format y values
    self.predict_df = self.create_graph_dataset(self.predict_df)
  
  def fit(self):
    train_loader = DataLoader([self.train_df], batch_size=64)

    loss_fn = nn.MSELoss()  # loss function used for regression
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)  # optimizer used for regression

    # train the model using the DataLoader object
    num_epochs = 500
    for _ in range(num_epochs):
        self.model.train()
        for data in train_loader:
            optimizer.zero_grad()
            output = self.model(data)
            loss = loss_fn(output, data.x)
            loss.backward()
            optimizer.step()
  
  def predict(self):

    self.model.eval()
    test_loader = DataLoader([self.predict_df], batch_size=64)
    for data in test_loader:
        output = self.model(data)
        y_pred = torch.split(output, 1, dim=1)[0]
        return y_pred.flatten().tolist()


class TimeSeriesGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(TimeSeriesGNN, self).__init__()
        torch.manual_seed(111)
        self.conv1 = GCNConv(in_channels, hidden_channels)  # first layer
        self.conv2 = GCNConv(hidden_channels, out_channels)  # second layer

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        # first GCN layer
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)

        # second GCN layer
        x = self.conv2(x, edge_index)
        
        return x
