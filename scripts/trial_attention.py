import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Add, Conv1D,Convolution2D, Bidirectional, LSTM, GRU, AlphaDropout, MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
from tensorflow_addons.layers import MultiHeadAttention
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn, json
import scipy.io as io
from typing import Any, Dict
from includes.clr_callback import *
path = '/home/maria/'
dataset_path = path + 'dataset_1d/'
with h5py.File(dataset_path +'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
print(X_train.shape)
with h5py.File(dataset_path +'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
    lbl_train = io.loadmat(dataset_path +'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path +'lbl_test.mat')['lbl_test']
print(lbl_test.shape)
print(lbl_train.shape)
Y_train = io.loadmat(dataset_path +'Y_train.mat')
Y_train = Y_train['Y_train']
print(Y_train.shape)
Y_test = io.loadmat(dataset_path +'Y_test.mat')
Y_test = Y_test['Y_test']
print(Y_test.shape)
classes = ['LFM', '2FSK', '4FSK', '8FSK', 'Costas', '2PSK', '4PSK', '8PSK', 'Barker', 'Huffman', 'Frank', 'P1', 'P2',
           'P3', 'P4', 'Px', 'Zadoff-Chu', 'T1', 'T2', 'T3', 'T4', 'NM', 'ruido']
AF = False
if AF:

    I_tr = X_train[:, :, 0]
    Q_tr = X_train[:, :, 1]
    X_tr = I_tr + 1j * Q_tr

    X_train[:, :, 1] = np.arctan2(Q_tr, I_tr) / np.pi
    X_train[:, :, 0] = np.abs(X_tr)

    I_te = X_test[:, :, 0]
    Q_te = X_test[:, :, 1]
    X_te = I_te + 1j * Q_te

    X_test[:, :, 1] = np.arctan2(Q_te, I_te) / np.pi
    X_test[:, :, 0] = np.abs(X_te)

    del I_tr
    del Q_tr
    del X_tr
    del I_te
    del Q_te
    del X_te

np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:],Y_train[:], lbl_train[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)
"""
class AttentionBlock():
    def init(self, 
             num_heads : int = 2,
             ff_dim = None,
             dropout: int = 0.1,
             key_dim : int = 2):
        self.attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)
        self.attention_dropout = keras.layers.Dropout(dropout)
        self.attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv1 = keras.layers.Conv1D(filters=ff_dim, kernel_size=1, activation='relu')
        self.dropout = keras.layers.Dropout(dropout)
        self.norm = keras.layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, input_shape):
        x_input = Input(input_shape)
        print("Input tensor", x_input)
        x = self.attention_layer(x_input, x_input)
        x = self.attention_dropout(x) 
        x = self.attention_norm(x)
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.norm(x)
        print("Output tensor", x)
        model = Model(inputs=x_input, outputs=x)
        model.summary()
        return model
        """

def ModelTrunk(input_shape : int):
    X_input = tf.keras.Input(input_shape)
    num_layers = 5
    attention_block = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
    x = attention_block(X_input,X_input)
    attention_dropout = keras.layers.Dropout(0.1)
    x= attention_dropout(x)
    attention_norm = keras.layers.LayerNormalization(epsilon=1e-6)
    x = attention_norm(x + X_input)
    conv1 = keras.layers.Conv1D(filters=None, kernel_size=1, activation='relu')
    x = conv1(x)
    dropout = keras.layers.Dropout(0.1)
    x = dropout(x)
    norm = keras.layers.LayerNormalization(epsilon=1e-6)
    x = norm(x + X_input)
    print(x)
   # x = Flatten()(x)
   # x = Dense(128, activation='selu')(x)
   # x = AlphaDropout(0.6)(x)
   # x = Dense(128, activation='selu')(x)
   # x = AlphaDropout(0.6)(x)
   # x = Dense(23, activation='softmax')(x)
    model = Model(inputs = X_input, outputs = x)
    model.summary()
    return model
model = ModelTrunk(X_train.shape[1:])
model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])
output_path = path + 'Results/model_stft'
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-4, step_size= 4 * (X_train.shape[0] // 256))
c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(X_train, Y_train, epochs = 500, batch_size = 256, callbacks = c, validation_data=(X_test, Y_test))
with open(output_path +'history_rnn.json', 'w') as f:
    json.dump(history.history, f)
model_json = model.to_json()
with open(output_path +'model_rnn.json', "w") as json_file:
    json_file.write(model_json)
