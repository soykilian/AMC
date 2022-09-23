import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Add, Conv1D,Convolution2D, Bidirectional, LSTM, GRU, AlphaDropout, MaxPooling1D
from tensorflow.keras.layers import MaxPooling2D, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import sklearn, json
import scipy.io as io
from typing import Any, Dict
import sys
path = '/home/maria/'
sys.path.insert(0, path + "AMC/includes")
from clr_callback import *
import matplotlib.pyplot as plt

dataset_path = path + 'Dataset_trials/'

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

classes = ['LFM', '2FSK', '4FSK', '8FSK', 'Costas', '2PSK', '4PSK', '8PSK', 'Barker', 'Huffman', 'Frank', 'P1', 'P2',
           'P3', 'P4', 'Px', 'Zadoff-Chu', 'T1', 'T2', 'T3', 'T4', 'NM', 'ruido']
AF = False
if AF:
    I_x = X_train[:, :, 0]
    Q_x = X_train[:, :, 1]
    X_train[:, :, 1] = np.arctan(Q_x, I_x) / np.pi
    X_train[:, :, 0] = np.abs(I_x + 1j * Q_x)
    I_t = X_test[:, :, 0]
    Q_t = X_test[:, :, 1]
    X_test[:, :, 1] = np.arctan(Q_t, I_t) / np.pi
    X_test[:, :, 0] = np.abs(I_t + 1j * Q_t)


np.random.seed(2022)
X_train, Y_train, lbl_train = sklearn.utils.shuffle(X_train[:], Y_train[:], lbl_train[:], random_state=2022)
X_val, Y_val, lbl_val = sklearn.utils.shuffle(X_val[:], Y_val[:], lbl_val[:], random_state=2022)
X_test, Y_test, lbl_test = sklearn.utils.shuffle(X_test[:], Y_test[:], lbl_test[:], random_state=2022)

print(X_train.shape)
print(X_val.shape)
print(X_test.shape)
print(lbl_train.shape)
print(lbl_val.shape)
print(lbl_test.shape)
print(Y_train.shape)
print(Y_val.shape)
print(Y_test.shape)

"""We have to create the structure of a residual stack"""

def res_stack(x, f):
    """
    x : input vector
    f : output shape
    """
    if x.shape[1] != f:
        x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x_shortcut = x
    x = Conv1D(f, 5, strides=1, padding="same", data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 5, strides=1, padding="same", data_format='channels_last')(x)
    x = BatchNormalization()(x)
    """Now we add the skip connection"""
    if x.shape[1] == x_shortcut.shape[1]:
        x = Add()([x, x_shortcut])
    else:
        raise Exception("Skip function not implemented")
    """After the addition the relu function is applied"""
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    return x


def ResNet(input_shape):
    X_input = Input(input_shape)
    ####
    #x = keras.layers.LayerNormalization(epsilon=1e-6)(x + X_input)
    #x = keras.layers.Conv1D(filters=2, kernel_size=5, padding="same",
            #activation='relu')(x)
   ### x = keras.layers.Dropout(0.1)(x)
    #x = keras.layers.LayerNormalization(epsilon=1e-6)(x + X_input)
    x = res_stack(X_input, 32)
    x = res_stack(x, 32)
    x = res_stack(x, 32)
    #x = res_stack(x, 32)
    #x = res_stack(x, 32)
    attention_block = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=2)
    x = attention_block(x, x, training=True)
    ####
    x = Flatten()(x)
    ####
    #x = Dense(128, activation='selu')(x)
    x = AlphaDropout(0.6)(x)
    ####
    #x = Dense(128, activation='selu')(x)
    #x = AlphaDropout(0.6)(x)
    ####
    x = Dense(23, activation='softmax')(x)
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = x)
    model.summary()
    return model

model = ResNet(X_train.shape[1:])
output_path = path + 'Results_attention'
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-4,
        step_size= 4 * (X_train.shape[0] // 256))
c = [clr_triangular, ModelCheckpoint(filepath= output_path +'/best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])


Train = True
if Train:
    history = model.fit(X_train, Y_train, epochs = 500, batch_size = 256, callbacks = c, validation_data=(X_val, Y_val))
    with open(output_path +'/history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(output_path +'/model_rnn.json', "w") as json_file:
        json_file.write(model_json)
else:
    model.load_weights(output_path +'/best_model.h5')
    with open(output_path +'/history_rnn.json', 'r') as f:
        history = json.load(f)


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'])
plt.show()
plt.savefig(output_path+ '/graphs/model_loss.pdf')


def getConfusionMatrixPlot(true_labels, predicted_labels,title):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
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


acc={}
snrs = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0,X_test.shape[0])))
    test_X_i = X_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
    test_Y_i = Y_test[[i for i,x in enumerate(test_SNRs) if x==snr]]

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    width = 18
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat, 1),title="ResNet Confusion Matrix (SNR=%d)"%(snr))
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(output_path + '/graphs/confmat_'+str(snr)+'.pdf')
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy: ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)
    with open(output_path + '/acc.json', 'w') as f:
        json.dump(acc, f)

plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on Radar Dataset")
plt.savefig(output_path + '/graphs/clas_acc.pdf')
