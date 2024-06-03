from keras.models import Sequential
from keras.layers import Dense, Activation





model = Sequential()
model.add(Dense(256, input_dim=5, activation='relu'))
model.add(Dense(128, input_dim=256, activation='relu'))
model.add(Dense(64, input_dim=128, activation='relu'))
model.add(Dense(32, input_dim=64, activation='relu'))
model.add(Dense(1, activation='softmax'))