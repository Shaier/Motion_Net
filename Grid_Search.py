#Hyperparameter search
#Hyperparameter search
#!pip install talos
#import talos as ta

p = {'activation':['relu', 'elu'],
     'optimizer' : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
     'losses': ['mse','rmse'],
     'batch_size': [10,20],
     'epochs': [10,20],
     'init_mode':['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
     'dropout_rate' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'learn_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
     'momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

}

parameters = {'batch_size': [10,20]}

t = ta.Scan([train1,train2,train3,train4,train5,train6,train7,train8,train9],
            [y1,y2,y3,y4,y5,y6,y7,y8,y9],
            params=p,
            model=model)
