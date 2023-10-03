import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from models.model import Model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger


class TransformerModel(Model):
  def __init__(self, train_df, predict_df) -> None:
    super().__init__(train_df, predict_df)

    # configure network and trainer
    early_stop_callback = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=10, verbose=False, mode='min')
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger('lightning_logs')  # logging results to a tensorboard

    self.model = pl.Trainer(
      max_epochs=12,
      accelerator='cpu',
      enable_model_summary=True,
      gradient_clip_val=0.1,
      limit_train_batches=50,  # coment in for training, running valiation every 30 batches
      # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
      callbacks=[lr_logger, early_stop_callback],
      logger=logger,
    )

  def preprocess_data(self):
    self.train_df['group_id'] = 0
    self.train_df['time_idx'] = [i for i in range(1, len(self.train_df) + 1)]  # TemporalFusionTransformer needs special index
    self.train_df['y'] = self.train_df['y'].astype(float)  # TemporalFusionTransformer works only with floats

    self.predict_df['group_id'] = 0
    self.predict_df['time_idx'] = [max(self.train_df['time_idx']) + i for i in range(1, len(self.predict_df) + 1)]  # TemporalFusionTransformer needs special index
    self.predict_df['y'] = self.predict_df['y'].fillna(0).astype(float)  # TemporalFusionTransformer works only with floats

    max_encoder_length = 7
    max_prediction_length = len(self.predict_df)
    self.predict_df = pd.concat([self.train_df.tail(max_encoder_length//2), self.predict_df])

    self.train_df = TimeSeriesDataSet(
        self.train_df,
        time_idx='time_idx',
        target='y',
        group_ids=['group_id'],
        min_encoder_length=max_encoder_length//2,
        max_encoder_length=max_encoder_length,
        min_prediction_length=1,
        max_prediction_length=max_prediction_length,
        static_categoricals=[],
        static_reals=[col for col in self.train_df.columns.tolist() if col not in ['ds', 'y', 'time_idx', 'group_id']],
        time_varying_known_categoricals=[],
        variable_groups={},
        time_varying_known_reals=[col for col in self.train_df.columns.tolist() if col not in ['ds', 'y', 'group_id']],
        time_varying_unknown_categoricals=[],
        time_varying_unknown_reals=[],
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
    )

    self.predict_df = TimeSeriesDataSet.from_dataset(
      self.train_df, # dataset from which to copy parameters (encoders, scalers, ...)
      self.predict_df,  # data from which new dataset will be generated
      predict=True, # predict the decoder length on the last entries in the time index
      stop_randomization=True,
    )
  
  def fit(self):
    tft = TemporalFusionTransformer.from_dataset(
      self.train_df,
      learning_rate=0.03,
      hidden_size=16,
      attention_head_size=2,
      dropout=0.1,
      hidden_continuous_size=8,
      loss=QuantileLoss(),
      log_interval=10,
      optimizer='Adam',
      reduce_on_plateau_patience=4,
    )

    batch_size = 128
    train_dataloader = self.train_df.to_dataloader(train=True, batch_size=batch_size)
    val_dataloader = self.predict_df.to_dataloader(train=False, batch_size=batch_size * 10)

    self.model.fit(
      tft,
      train_dataloaders=train_dataloader,
      val_dataloaders=val_dataloader,
    )
  
  def predict(self):
    best_model_path = self.model.checkpoint_callback.best_model_path
    best_tft = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
    batch_size = 128
    val_dataloader = self.predict_df.to_dataloader(train=False, batch_size=batch_size * 10)
    predictions = best_tft.predict(val_dataloader)
    return predictions.tolist()[0]
