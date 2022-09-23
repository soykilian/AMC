import h5py
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import layers, optimizers, activations, initializers,regularizers, constraints
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D,Convolution2D, MaxPooling1D, AlphaDropout, Layer, LSTM, Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
#from tensorflow_addons.layers import MultiHeadAttention
import numpy as np
import tensorflow.keras as keras
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn, json
import scipy.io as io
from typing import Any, Dict
import logging, sys
from attention import Attention
logging.disable(sys.maxsize)
path = '/home/maria/'
sys.path.insert(0, path + "AMC/includes")
from clr_callback import *
import matplotlib.pyplot as plt


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#sess = tf.Session(config=config)

#classes = ['LFM','2FSK','4FSK','8FSK', 'Costas','2PSK','4PSK','8PSK','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4','NM','ruido']
classes = ['LFM', 'BFSK', 'BPSK', 'NM', 'LFM_ESC', 'SIN', 'BASK']
dt = np.dtype(float)
dataset_path = path + 'Dataset/'

with h5py.File(dataset_path +'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
with h5py.File(dataset_path +'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
with h5py.File(dataset_path +'X_val.mat', 'r') as f:
    X_val = np.array(f['X_val']).T
lbl_train = io.loadmat(dataset_path + 'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path + 'lbl_test.mat')['lbl_test']
lbl_val = io.loadmat(dataset_path + 'lbl_val.mat')['lbl_val']

Y_val = io.loadmat(dataset_path + 'Y_val.mat')
Y_val = Y_val['Y_val']
Y_train = io.loadmat(dataset_path + 'Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = io.loadmat(dataset_path + 'Y_test.mat')
Y_test = Y_test['Y_test']

print("X test shape: ", X_test.shape)
print("X val shape: ", X_val.shape)
print("Y train shape: ", Y_train.shape)
print("Y val shape: ", Y_val.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label val shape: ", lbl_val.shape)
print("Label test shape: ", lbl_test.shape)

AP = False

if AP:
    I_tr = X_train[:,:,0]
    Q_tr = X_train[:,:,1]
    X_tr = I_tr+ 1j*Q_tr
    
    X_train[:,:,1] = np.arctan2(Q_tr, I_tr)/np.pi
    X_train[:,:,0] = np.abs(X_tr)
    
    I_te = X_test[:,:,0]
    Q_te = X_test[:,:,1]
    X_te = I_te+ 1j*Q_te
    
    X_test[:,:,1] = np.arctan2(Q_te, I_te)/np.pi
    X_test[:,:,0] = np.abs(X_te)
    
    del I_tr
    del Q_tr
    del X_tr
    del I_te
    del Q_te
    del X_te

np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:], Y_train[:], lbl_train[:], random_state=2022)
X_val, Y_val, lbl_val = sklearn.utils.shuffle(X_val[:], Y_val[:], lbl_val[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)

def DenseNet(input_shape):
    x_input = Input(input_shape)
    x_input = keras.backend.transpose(x_input[-1])
    x_input = Input((x_input[0], x_input[1]))
    x = Dense(512,input_shape=(input_shape[1], input_shape[0]),name ='dense_0')(x_input)
    x = Dense(256, name ='dense_1')(x)
    x = Dense(128, name ='dense_2')(x)
    x = Dense(7, activation='softmax',name ='classification')(x)
    model = Model(inputs=x_input, outputs=x)
    model.summary()
    return model

model = DenseNet(X_train.shape[1:])


