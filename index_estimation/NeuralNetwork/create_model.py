from keras.layers import Dense
from keras import models
from keras.layers import Dropout
import numpy as np

def lac_create_model(n_features,activation , optimizer):
    
    model = models.Sequential()
    model.add(Dense(units=64,activation='linear',input_shape=(n_features,)))
    model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=256, activation=activation))
    #model.add(Dense(units=512, activation=activation))
    #model.add(Dense(units=1024, activation=activation))
    model.add(Dense(units=512, activation=activation))
    model.add(Dense(units=256, activation=activation))
    #model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=64, activation=activation))
    model.add(Dense(units=32, activation=activation))
    model.add(Dense(units=16, activation=activation))
    #model.add(Dense(units=4, activation='linear'))
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(optimizer=optimizer , loss='MSE', metrics=['mae'])
    
    return model

def lac_create_model_result(activation , optimizer):
    
    model = models.Sequential()
    model.add(Dense(units=64,activation='linear'))
    model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=256, activation=activation))
    #model.add(Dense(units=512, activation=activation))
    #model.add(Dense(units=1024, activation=activation))
    model.add(Dense(units=512, activation=activation))
    model.add(Dense(units=256, activation=activation))
    model.add(Dense(units=128, activation=activation))
    model.add(Dense(units=64, activation=activation))
    model.add(Dense(units=32, activation=activation))
    model.add(Dense(units=16, activation=activation))
    #model.add(Dense(units=4, activation='linear'))
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(optimizer=optimizer , loss='MSE', metrics=['mae'])
    
    return model

def lac_create_model_charenge(activation , optimizer):
    
    model = models.Sequential()
    model.add(Dense(units=512, activation=activation))
    model.add(Dense(units=256, activation=activation))
    model.add(Dense(units=64, activation=activation))
    model.add(Dense(units=32, activation=activation))
    model.add(Dense(units=1, activation='linear'))
    
    model.compile(optimizer=optimizer , loss='MSE', metrics=['mae'])
    
    return model


#model = lac_create_model_result(activation='relu', optimizer='adamax')
#dummy_input = np.random.random((1, 100))
#model(dummy_input)
#model.summary()