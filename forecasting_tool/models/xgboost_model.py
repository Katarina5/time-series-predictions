from models.forecaster_model import ForecasterModel


class XGBoostModel(ForecasterModel):
    def __init__(self, train_df, predict_df) -> None:
        super().__init__(train_df, predict_df, 'xgboost')
