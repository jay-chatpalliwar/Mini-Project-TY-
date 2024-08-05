import os
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf

import preprocess

X_train = preprocess.X_train
y_train = preprocess.y_train

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

actions = np.array(['hello', 'thanks', 'yes','no','drink','please','good_luck','help','congratulations','hungry'])

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=250, callbacks=[tb_callback])


model.save('action1.h5')