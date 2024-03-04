arima = {
    'order':[(2,1,0),(0,1,2),(1,1,1),(2,0,0)],
    'seasonal_order':[(0,1,1,7),(2,1,0,7),(2,0,0,7),
                      (0,1,1,10),(2,1,0,10),(2,0,0,10)]
}

catboost = {
    'iterations': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'depth': [4, 6, 8],
    'l2_leaf_reg': [1, 3, 5, 7, 9],
    'verbose': [0]
}

lstm = {
    'lstm_layer_sizes':[(50,50,50)],
    'activation':['relu','tanh'],
    'dropout':[(0,0,0),(.2,.2,.2),],
    'lags':[10,12,25,50],
    'verbose':[0],
    'epochs':[5,10,25,50]
}

rnn = {
    'layers_struct':[
        [
            ('LSTM',{'units':50,'activation':'relu'}),
            ('LSTM',{'units':50,'activation':'relu'}),
            ('LSTM',{'units':50,'activation':'relu'}),
        ],
        [
            ('LSTM',{'units':50,'activation':'tanh'}),
            ('LSTM',{'units':50,'activation':'tanh'}),
            ('LSTM',{'units':50,'activation':'tanh'}),
        ],
        [
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
            ('LSTM',{'units':50,'activation':'tanh','dropout':.2}),
        ],
    ],
    'epochs':[25],
    'verbose':[0]
}

svr = {
    'kernel':['linear','rbf','sigmoid','poly'],
    'C':[.01,.1,.5,1,2,3],
    'epsilon':[0.01,0.1,0.5,1],
    'gamma':['scale','auto']
}

xgboost = {
     'n_estimators':[150,200,250],
     'scale_pos_weight':[5,10],
     'learning_rate':[0.1,0.2],
     'gamma':[0,3,5],
     'subsample':[0.8,0.9]
}
