#Hyperparameter search


from keras.wrappers.scikit_learn import KerasRegressor

parameters = {'activation':['relu', 'elu'],
     'optimizer' : ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'],
     'losses': ['mse','rmse'],
     'batch_size': [10,20],
     'epochs': [10,20],
     'init_mode':['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform'],
     'dropout_rate' : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
     'learn_rate' : [0.001, 0.01, 0.1, 0.2, 0.3],
     'momentum' : [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]

}

p = {'batch_size': [10,20]}

from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout

def create_model(optimizer='adam',loss='mse',learn_rate=0.00, momentum=0,init_mode='uniform'): #added for grid search
    # Multiple Inputs

    #Idea here: Get the weights, run the new frames on the weights, if the loss<threshhold == class 1

    # 1st input model
    frame1 = Input(shape=(9216,))
    hidden1 = Dense(30, kernel_initializer=init_mode, activation='relu')(frame1)
    #hidden1= Dropout(dropout_rate)(hidden1)
    hidden1 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden1)
    #hidden1 = Dense(30, activation='relu')(hidden1)
    output1 = Dense(9216, activation='linear')(hidden1) #frame 2 is output 1

    # 2nd input model
    frame2 = Input(shape=(9216,))
    hidden2 = Dense(30, kernel_initializer=init_mode,activation='relu')(frame2)
    hidden2 = Dense(50,kernel_initializer=init_mode, activation='relu')(hidden2)
    #hidden2 = Dense(30, activation='relu')(hidden2)
    output2 = Dense(9216, activation='linear')(hidden2) #frame 3 is output 2

    # 3rd input model
    frame3 = Input(shape=(9216,))
    hidden3 = Dense(30, kernel_initializer=init_mode,activation='relu')(frame3)
    hidden3 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden3)
    #hidden3 = Dense(30, activation='relu')(hidden3)
    output3 = Dense(9216, activation='linear')(hidden3) #frame 4 is output 3

    # 4th input model
    frame4 = Input(shape=(9216,))
    hidden4 = Dense(30, kernel_initializer=init_mode,activation='relu')(frame4)
    hidden4 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden4)
    #hidden4 = Dense(30, activation='relu')(hidden4)
    output4 = Dense(9216, activation='linear')(hidden4) #frame 5 is output 4

    # 5th input model
    frame5 = Input(shape=(9216,))
    hidden5 = Dense(30,kernel_initializer=init_mode, activation='relu')(frame5)
    hidden5 = Dense(50,kernel_initializer=init_mode, activation='relu')(hidden5)
    #hidden5 = Dense(30, activation='relu')(hidden5)
    output5 = Dense(9216, activation='linear')(hidden5) #frame 6 is output 5

    # 6th input model
    frame6 = Input(shape=(9216,))
    hidden6 = Dense(30,kernel_initializer=init_mode, activation='relu')(frame6)
    hidden6 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden6)
    #hidden6 = Dense(30, activation='relu')(hidden6)
    output6 = Dense(9216, activation='linear')(hidden6) #frame 7 is output 6

    # 7th input model
    frame7 = Input(shape=(9216,))
    hidden7 = Dense(30, kernel_initializer=init_mode,activation='relu')(frame7)
    hidden7 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden7)
    #hidden7 = Dense(30, activation='relu')(hidden7)
    output7 = Dense(9216, activation='linear')(hidden7) #frame 8 is output 7

    # 8th input model
    frame8 = Input(shape=(9216,))
    hidden8 = Dense(30,kernel_initializer=init_mode, activation='relu')(frame8)
    hidden8 = Dense(50, kernel_initializer=init_mode,activation='relu')(hidden8)
    #hidden8 = Dense(30, activation='relu')(hidden8)
    output8 = Dense(9216, activation='linear')(hidden8) #frame 9 is output 8

    # 9th input model
    frame9 = Input(shape=(9216,))
    hidden9 = Dense(30,kernel_initializer=init_mode, activation='relu')(frame9)
    hidden9 = Dense(50,kernel_initializer=init_mode, activation='relu')(hidden9)
    #hidden9 = Dense(30, activation='relu')(hidden9)
    output9 = Dense(9216, activation='linear')(hidden9) #frame 10 is output 9

    model = Model(inputs=[frame1, frame2, frame3,frame4, frame5, frame6,frame7, frame8, frame9], outputs=[output1, output2, output3,output4, output5, output6, output7,output8, output9])
    model.compile(optimizer=optimizer, loss=loss, metrics=['mse'])
    return model #added for grid search

model = KerasRegressor(build_fn=create_model)
# summarize layers
#print(model.summary())
# plot graph
#plot_model(model, to_file='model.png')


#grid search
grid = GridSearchCV(estimator=model, param_grid=p, n_jobs=1)
grid_result = grid.fit(X=[train1,train2,train3,train4,train5,train6,train7,train8,train9],
          y=[y1,y2,y3,y4,y5,y6,y7,y8,y9])

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
