from lstm_embedding import embedding
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

features = 300

class Smart_Batch_Padding(tf.keras.layers.Layer):
  def call(self, inputs, mask):

    seq_lengths = tf.math.count_nonzero(mask,axis=1)
    max_seq_len = tf.math.reduce_max(seq_lengths)
    return inputs[:,:max_seq_len,:], mask[:,:max_seq_len]

class LstmClassifier(tf.keras.Model):
  
  def __init__(self, dense_layers=1, lstm_cells=128):

    super(LstmClassifier, self).__init__()

    self.batchPadSlicing = Smart_Batch_Padding()
    self.mask1 = tf.keras.layers.Masking(mask_value=0.,input_shape=(None, features))
    self.lstm1 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_cells, dropout=0.2, return_sequences=True), input_shape=(None, features), name='bidirectional_1')
    self.lstm2 = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(lstm_cells, dropout=0.2, return_sequences=True), name='bidirectional_2')
    self.attentionDense = tf.keras.layers.Dense(1, activation='tanh', name = "attention_dense")
    self.softmax = tf.keras.layers.Softmax(axis=1, name="softmax")
    self.dropout = tf.keras.layers.Dropout(0.2, name="dropout")
    if dense_layers == 3:
      self.dense2 = tf.keras.layers.Dense(300, activation='relu', name='dense_2')
      self.dense1 = tf.keras.layers.Dense(100, activation='relu', name='dense_1')

    if dense_layers == 2:
      self.dense1 = tf.keras.layers.Dense(100, activation='relu', name='dense_1')
    
    self.outputLayer = tf.keras.layers.Dense(24, name='output',activation='sigmoid')

  def attentionPooling(self,x,mask):
    x = self.mask1(x)
    temp = self.attentionDense(x)[:,:,0]
    attentions = self.softmax(temp, mask = mask)
    x = tf.transpose(x, perm=[0,2,1])
    weightedSum = tf.linalg.matvec(x, attentions)
    return weightedSum

  def call(self, inputs):
    mask = self.mask1(inputs)._keras_mask
    inputs, mask = self.batchPadSlicing(inputs, mask)
    x = self.lstm1(inputs, mask = mask)
    x = self.lstm2(x, mask = mask)
    x = self.attentionPooling(x, mask = mask)
    x = self.dropout(x)

    if self.dense_layers == 3:
      x = self.dense2(x)
      x = self.dropout(x)
      x = self.dense1(x)
      x = self.dropout(x)

    elif self.dense_layers == 2:
      x = self.dense1(x)
      x = self.dropout(x)
    
    return self.outputLayer(x)

def make_dataset_df(filepath: str, sep=','):
  
  
  return df


def make_dataset(dataset_df):
  X = embedding(dataset_df[dataset_df.columns[-1]].to_numpy())
  target_cols = dataset_df.columns[1:-1]
  y = tf.constant(dataset_df[target_cols].to_numpy())
  return X, y

X = np.load('data/x_test.npy',allow_pickle=True)
y = np.load('data/y_test.npy',allow_pickle=True)

X = pad_sequences(
    X, padding="post", dtype=float)
maxTrainingLength = len(X[0])
X = np.reshape(X,(-1, maxTrainingLength, features))
print(y[0])
X = tf.constant(X)
y = tf.constant(y)

model = LstmClassifier(2,256)
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy','AUC','binary_accuracy'], run_eagerly=True)

model(X[0])
model.load_weights("models/bilstm.h5")

# predict with model
y_pred = model.predict(X)
y_pred[y_pred <= 0.5] = 0.0
y_pred[y_pred > 0.5] = 1.0
y_pred = y_pred.astype(int)

# load again test csv
df = pd.read_csv("data/absita_2018_test.csv", sep=';')
target_cols = df.columns[1:-1]

# assign predicted values to target columns
df_pred = pd.DataFrame(data=y_pred, columns = target_cols)
df[target_cols] = df_pred

# save results in csv using ; (required by the evaluation script)
df.to_csv("data/absita_2018_baseline.csv", index=False, sep=';')