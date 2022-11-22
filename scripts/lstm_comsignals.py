import h5py
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, activations, initializers,regularizers, constraints
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D,Convolution2D, MaxPooling1D, AlphaDropout,Layer, LSTM, Layer
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers
import tensorflow as tf
import tensorflow.keras.backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
path = '/home/maria/'
sys.path.insert(0, path + "AMC/includes")
from clr_callback import *
import json

import scipy.signal as sc
from sklearn.metrics import confusion_matrix
import cmath
import pickle

X = np.load('/home/maria/X_shuffled.npy')
Y = np.load('/home/maria/Y_shuffled.npy')
SNR = np.load('/home/maria/SNR_shuffled.npy')

classes = ['OOK', '4ASK', '8ASK','BPSK', 'QPSK', '8PSK', '16PSK', '32PSK', '16APSK', '32APSK', '64APSK', '128APSK','16QAM', '32QAM', '64QAM', '128QAM', '256QAM', 'AM-SSB-WC', 'AM-SSB-SC', 'AM-DSB-WC', 'AM-DSB-SC', 'FM', 'GMSK', 'OQPSK']

snrs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

n_resize = X.shape[0]
X_red = X[:n_resize]
Y_red = Y[:n_resize]
n_examples = X_red.shape[0]
n_train = int(n_examples * 0.5)
n_test = int(n_examples * 0.25)

X_train = X_red[:n_train]
Y_train = Y_red[:n_train]

X_val =  X_red[n_train:n_train+n_test]
Y_val =  Y_red[n_train:n_train+n_test]

X_test = X_red[n_train+n_test:]
Y_test = Y_red[n_train+n_test:]
print(X_train.shape)
print(X_val.shape)
print(X_test.shape)

AP = False 
if AP:
    for i in range(X_train.shape[0]):
        X_train_cmplx = X_train[i,:,0] + 1j* X_train[i,:,1]
        
        X_train_ang = np.arctan2(X_train[i,:,1],X_train[i,:,0])/np.pi
        X_train_amp = np.abs(X_train_cmplx)
        
        X_train_f[i,:,0] =  (X_train[i,:,0] - np.min(X_train[i,:,0])) / (np.max(X_train[i,:,0]) - np.min(X_train[i,:,0]))
        X_train_f[i,:,1] =  (X_train[i,:,1] - np.min(X_train[i,:,1])) / (np.max(X_train[i,:,1]) - np.min(X_train[i,:,1]))
        X_train_f[i,:,2] = (X_train_amp - np.min(X_train_amp)) / (np.max(X_train_amp) - np.min(X_train_amp))
        X_train_f[i,:,3] = X_train_ang

    for i in range(X_test.shape[0]):
        X_test_cmplx = X_test[i,:,0] + 1j* X_test[i,:,1]
        
        X_test_ang = np.arctan2(X_test[i,:,1],X_test[i,:,0])/np.pi
        X_test_amp = np.abs(X_test_cmplx)
        
        X_test_f[i,:,0] =  (X_test[i,:,0] - np.min(X_test[i,:,0])) / (np.max(X_test[i,:,0]) - np.min(X_test[i,:,0]))
        X_test_f[i,:,1] =  (X_test[i,:,1] - np.min(X_test[i,:,1])) / (np.max(X_test[i,:,1]) - np.min(X_test[i,:,1]))
        X_test_f[i,:,2] = (X_test_amp - np.min(X_test_amp)) / (np.max(X_test_amp) - np.min(X_test_amp))
        X_test_f[i,:,3] = X_test_ang


# In[5]:


def RecComModel(input_shape):
    x_input = Input(input_shape)
    x = LSTM(128, return_sequences=True, name='lstm0')(x_input)
    x = LSTM(128, return_sequences=True, name='lstm1')(x)
    x = LSTM(128, return_sequences=False, name='lstm2')(x)
    x = Dense(24, activation='softmax', name='fc0')(x)
    model = Model(inputs = x_input, outputs = x)
    model.summary()
    return model


output_path = path + 'Results_lstm/com_signals'
model = RecComModel(X_train.shape[1:])
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-6, max_lr=1e-3,
        step_size= 4 * (X_train.shape[0] // 256))

c=[clr_triangular, ModelCheckpoint(filepath= output_path +'/best_model.h5',
    monitor='val_loss', save_best_only=True)]
model.compile(optimizer=optimizers.Adam(1e-6), loss='categorical_crossentropy', metrics=['accuracy'])


Train = True
if Train:
    #tuner.search(X_train, Y_train, epochs=100, ba)
    history = model.fit(X_train, Y_train, epochs = 300, batch_size = 256, callbacks = c, validation_data=(X_val, Y_val))
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
plt.savefig(output_path+ '/graphs/model_loss.pdf')


def getConfusionMatrixPlot(true_labels, predicted_labels,title):
    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)

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
signal_class = {classes[0]: np.zeros(26), classes[1]:
        np.zeros(26),classes[2]:np.zeros(26), classes[3] : np.zeros(26),
        classes[4]: np.zeros(26), classes[5]:np.zeros(26),
        classes[6]:np.zeros(26), classes[7]:np.zeros(26),
        classes[8]:np.zeros(26), classes[9]:np.zeros(26),
        classes[10]:np.zeros(26), classes[11]:np.zeros(26),
        classes[12]:np.zeros(26), classes[13]:np.zeros(26),
        classes[14]:np.zeros(26), classes[15]:np.zeros(26),
        classes[16]:np.zeros(26),  classes[17]:np.zeros(26),
        classes[18]:np.zeros(26), classes[19]:np.zeros(26),
        classes[20]:np.zeros(26), classes[21]:np.zeros(26),
        classes[22]:np.zeros(26), classes[23]:np.zeros(26)}
acc={}
snrs = [-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]

for j,snr in enumerate(snrs):
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0,X_test.shape[0])))
    test_X_i = X_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
    test_Y_i = Y_test[[i for i,x in enumerate(test_SNRs) if x==snr]]

    # estimate classes
    test_Y_i_hat = np.array(model.predict(test_X_i))
    cm = confusion_matrix(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat,1))
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)
    cm = np.round(cm_norm,2)
    for i in range(len(classes)):
        signal_class[classes[i]][j]= cm[i][i]
    width = 18
    height = width / 1.618
    plt.figure(figsize=(width, height))
    plt = getConfusionMatrixPlot(np.argmax(test_Y_i, 1), np.argmax(test_Y_i_hat,
        1),title="Attention Confusion Matrix (SNR=%d)"%(snr))
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

plt.figure()
for i in range(len(classes)):
    plt.plot(snrs, signal_class[classes[i]])
plt.legend(classes)
plt.xlabel("Signal to Noise Ratio")
