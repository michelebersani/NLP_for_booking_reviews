import numpy as np
import pickle
import tensorflow as tf
from tensorflow import keras

modelName = "2_layers_biLSTM"

xTrain = np.load('data/x_train.npy',allow_pickle=True)
yTrain = np.load('data/y_train.npy',allow_pickle=True)

xTrain = keras.preprocessing.sequence.pad_sequences(
    xTrain, padding="post", dtype=float)
maxTrainingLength = len(xTrain[0])
features = 300
xTrain = np.reshape(xTrain,(-1, maxTrainingLength, features))

xVal = np.load('data/x_val.npy',allow_pickle=True)
xVal = keras.preprocessing.sequence.pad_sequences(
    xVal, padding="post", dtype=float)
maxValLength = len(xVal[0])
features = 300
xVal = np.reshape(xVal,(-1, maxValLength, features))
yVal = np.load('data/y_val.npy',allow_pickle=True)

model = keras.Sequential([
    keras.layers.Masking(mask_value=0.,input_shape=(None, features)),
    keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2, return_sequences=True), input_shape=(None, 300), name='bidirectional_1'),
    keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2), name='bidirectional_2'),
    keras.layers.Dense(300, activation='relu', name='dense_1'),
    keras.layers.Dense(100, activation='relu', name='dense_2'),
    keras.layers.Dense(24, name='output')
])
model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])

history = model.fit(x = xTrain, y = yTrain,
                    batch_size=10,
                    epochs=10,
                    validation_data=(xVal,yVal)
                )

with open(f'trained_models/{modelName}_history', 'wb') as file:
        pickle.dump(history.history, file)
model.save(f'trained_models/{modelName}')