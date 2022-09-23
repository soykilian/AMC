#/**************************IMPORTS****************************/

import h5py
import numpy as np
import sys
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D,Convolution2D, MaxPooling1D, AlphaDropout
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
logging.disable(sys.maxsize)
path = '/home/maria/'
sys.path.insert(0, path + "AMC/includes")
from clr_callback import *
import matplotlib.pyplot as plt

#/**************************DATASET****************************/

dataset_path = path + 'Dataset/'
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

#/***********************************************************/
# INNERPHASE & QUADRATURE --> MODULE & PHASE
# Change in c
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

#/*********************************************************/
# Shuffle the data

np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:],Y_train[:], lbl_train[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)

# Organize in classes for future work once the first version works
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

#/*********************************************************/
# Model definition
ACTIVATION = 'selu'
HIDDEN = 32
INITIALIZER = 'lecun_normal'

def ModelTrunk(input_shape : int):
    X_input = tf.keras.Input(input_shape)
    conv = Conv1D(2, 5, strides=1, activation=ACTIVATION, padding="same",
            data_format='channels_last')(X_input)
    conv = BatchNormalization()(conv)
    conv = Conv1D(2, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(2, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = BatchNormalization()(conv)
    conv = MaxPooling1D(2)(conv)
    conv = BatchNormalization()(conv)
    conv =Conv1D(2, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = BatchNormalization()(conv)
    conv = Conv1D(2, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = BatchNormalization()(conv)
    conv = keras.layers.MaxPooling1D(2)(conv)
    conv = BatchNormalization()(conv)
    attention_block = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
    x = attention_block(conv, conv)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + conv)
    x = keras.layers.Conv1D(filters=2, kernel_size=5, padding="same",
            activation='relu')(x)
    x = keras.layers.Dropout(0.1)(x)
    x = keras.layers.LayerNormalization(epsilon=1e-6)(x + conv)
    
    """
    conv = Conv1D(HIDDEN, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(x)
    #print("------------------------------CONV--------------------------------------------")
    conv = Conv1D(HIDDEN, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = MaxPooling1D(2)(conv)
    conv =Conv1D(HIDDEN, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = Conv1D(HIDDEN, 5, strides=1, activation=ACTIVATION, padding="same", data_format='channels_last')(conv)
    conv = keras.layers.MaxPooling1D(2)(conv)
    """
    # recurrent with attention, this generates sequences
    """
    recurrent_forward = LSTM(HIDDEN, return_sequences=True, name= 'lstm0')(conv)
    print(recurrent_forward)
    recurrent_backward =  LSTM(HIDDEN, return_sequences=True, name= 'lstm0')(TimeStepReverse()(conv))
    print(recurrent_backward)
    recurrent = keras.layers.Concatenate()(
        [recurrent_forward, recurrent_backward])
    """
    # Try wth 3 attention blocks at least for improvement
    x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    x = Dense(256, activation=ACTIVATION)(x)
    x = AlphaDropout(0.6)(x)
    x = Dense(23, activation='softmax')(x)
    model = Model(inputs = X_input, outputs = x)
    model.summary()
    return model
#/*********************************************************/
#Model initialization and compilation with ciclical callbacks

model = ModelTrunk(X_train.shape[1:])
model.compile(optimizer=optimizers.Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])
output_path = path + 'Results_attention/'
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-6, max_lr=1e-3, step_size= 4 * (X_train.shape[0] // 256))
c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]
history = model.fit(X_train, Y_train, epochs = 20, batch_size = 256, callbacks = c, validation_data=(X_test, Y_test))
with open(output_path +"history_rnn.json", 'w') as f:
    json.dump(history.history, f)
model_json = model.to_json()
with open(output_path +"model_rnn.json", 'w') as json_file:
    json_file.write(model_json)

model.load_weights(output_path + 'best_model.h5')

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'])
plt.show()
plt.savefig(output_path + "graphs/model_loss.pdf")


# In[ ]:


def getConfusionMatrixPlot(true_labels, predicted_labels, title):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm, 2)
    print(cm)

    # create figure
    width = 18
    height = width / 1.618
    fig = plt.figure(figsize=(width, height))
    plt.clf()
    ax = fig.add_subplot(111)
    ax.set_aspect(1)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    res = ax.imshow(cm, cmap=plt.cm.binary,
                    interpolation='nearest', vmin=0, vmax=1)

    # add color bar
    plt.colorbar(res)

    # annotate confusion entries
    width = len(cm)
    height = len(cm[0])

    for x in range(width):
        for y in range(height):
            ax.annotate(str(cm[x][y]), xy=(y, x), horizontalalignment='center',
                        verticalalignment='center', color=getFontColor(cm[x][y]))

    # add genres as ticks
    alphabet = classes
    plt.xticks(range(width), alphabet[:width], rotation=30)
    plt.yticks(range(height), alphabet[:height])
    plt.title(title)
    return plt

def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"

acc = {}
snrs = [-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0, X_test.shape[0])))
    test_X_i = X_test[[i for i, x in enumerate(test_SNRs) if x == snr]]
    test_Y_i = Y_test[[i for i, x in enumerate(test_SNRs) if x == snr]]

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 18
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1),
                                 title="ResNet Confusion Matrix (SNR=%d)" % (snr))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(output_path + "graphs/confmat_" + str(snr) + '.pdf')
    conf = np.zeros([len(classes), len(classes)])
    confnorm = np.zeros([len(classes), len(classes)])
    for i in range(0, test_X_i.shape[0]):
        j = list(test_Y_i[i, :]).index(1)
        k = int(np.argmax(test_Y_i_hat[i, :]))
        conf[j, k] = conf[j, k] + 1
    for i in range(0, len(classes)):
        confnorm[i, :] = conf[i, :] / np.sum(conf[i, :])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor + ncor))
    acc[snr] = 1.0 * cor / (cor + ncor)
# print(acc)


# In[ ]:


with open(output_path + 'acc.json', 'w') as f:
    json.dump(acc, f)

plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on Radar Dataset")
plt.savefig(output_path + "graphs/clas_acc.pdf")

