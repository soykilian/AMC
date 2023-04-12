import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, activations, initializers,regularizers, constraints
from tensorflow.keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv1D,Convolution2D, MaxPooling1D, AlphaDropout,Layer, LSTM, Layer
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
import keras_tuner
logging.disable(sys.maxsize)
path = '/home/maria/'
sys.path.insert(0, path + "AMC/includes")
from clr_callback import *
import matplotlib.pyplot as plt


#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#sess = tf.Session(config=config)
classes = ['LFM','2FSK','4FSK','8FSK', 'Costas','2PSK','4PSK','8PSK','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4','NM','ruido'] 
#classes = ['LFM', 'BFSK', 'BPSK', 'NM', 'LFM_ESC', 'SIN', 'EXP', 'BASK']
dataset_path = path + 'Dataset_23corr/'

with h5py.File(dataset_path +'Corr_train.mat', 'r') as f:
    X_train = np.array(f['Corr_train'], dtype='float32').T
with h5py.File(dataset_path +'Corr_test.mat', 'r') as f:
    X_test = np.array(f['Corr_test'], dtype='float32').T
with h5py.File(dataset_path +'Corr_val.mat', 'r') as f:
   X_val = np.array(f['Corr_val'], dtype='float32').T

Y_val = io.loadmat(dataset_path + 'Y_val.mat')
Y_val = Y_val['Y_val']
Y_train = io.loadmat(dataset_path + 'Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = io.loadmat(dataset_path + 'Y_test.mat')
Y_test = Y_test['Y_test']
lbl_train = io.loadmat(dataset_path + 'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path + 'lbl_test.mat')['lbl_test']
lbl_val = io.loadmat(dataset_path + 'lbl_val.mat')['lbl_val']

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
X_train, Y_train,lbl_train = sklearn.utils.shuffle(X_train[:], Y_train[:] ,lbl_train[:],random_state=2022)

X_val, Y_val , lbl_val = sklearn.utils.shuffle(X_val[:], Y_val[:], lbl_val[:],random_state=2022)
X_test, Y_test , lbl_test= sklearn.utils.shuffle(X_test[:], Y_test[:],lbl_test[:] ,random_state=2022)


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)
print(X_test.shape[0])
def RecComModel():
    x_input = Input([1024,2])
    x = LSTM(128, return_sequences=True, name='lstm0')(x_input)
    x = LSTM(128, return_sequences=True, name='lstm1')(x)
    x = LSTM(128, return_sequences=False, name='lstm2')(x)
    x = Dense(23, activation='softmax', name='fc0')(x)
    model = Model(inputs = x_input, outputs = x)
    model.summary()
    return model


output_path = path + '23_signals/corr'
model = RecComModel()
clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-3, step_size= 4 * (X_train.shape[0] // 500))
c=[clr_triangular, ModelCheckpoint(filepath= output_path +'/best_model.h5', monitor='val_loss', save_best_only=True)]
model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])


Train = True
if Train:
    history = model.fit(X_train,Y_train, epochs = 300, batch_size = 500, callbacks = c, validation_data=(X_val, Y_val))
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
signal_class = {classes[0]: np.zeros(17), classes[1]:
        np.zeros(17),classes[2]:np.zeros(17), classes[3] : np.zeros(17),
        classes[4]: np.zeros(17), classes[5]:np.zeros(17),
        classes[6]:np.zeros(17), classes[7]:np.zeros(17),
        classes[8]:np.zeros(17), classes[9]:np.zeros(17),
        classes[10]:np.zeros(17), classes[11]:np.zeros(17),
        classes[12]:np.zeros(17), classes[13]:np.zeros(17),
        classes[14]:np.zeros(17), classes[15]:np.zeros(17),
        classes[16]:np.zeros(17),  classes[17]:np.zeros(17),
        classes[18]:np.zeros(17), classes[19]:np.zeros(17),
        classes[20]:np.zeros(17), classes[21]:np.zeros(17), classes[22]:np.zeros(17)}
acc={}
snrs = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]

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
