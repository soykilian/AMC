#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Input, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, Conv1D, Convolution2D, Bidirectional, LSTM, GRU, CuDNNLSTM, MaxPooling1D, Add, AlphaDropout
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Dropout, GlobalMaxPooling2D, GlobalAveragePooling2D, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from tensorflow.keras.utils import plot_model

import tensorflow as tf
from tensorflow.keras import backend as K
K.set_image_data_format('channels_last')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import json

import scipy.signal as sc
from sklearn.metrics import confusion_matrix
import cmath
import pickle
import scipy.io as sio
import h5py

from tensorflow.keras.callbacks import *
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../CLR')
from clr_callback import *


# In[2]:
#from keras.backend.tensorflow_backend import set_session

#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
#sess = tf.Session(config=config)

classes = ['LFM','2FSK','4FSK','8FSK', 'Costas','2PSK','4PSK','8PSK','Barker','Huffman','Frank','P1','P2','P3','P4','Px','Zadoff-Chu','T1','T2','T3','T4','NM','ruido']
dt = np.dtype(float)

with h5py.File('../Datasets/radar/Interpolation_orthogonal/X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
with h5py.File('../Datasets/radar/Interpolation_orthogonal/X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
Y_train = sio.loadmat('../Datasets/radar/Interpolation_orthogonal/Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = sio.loadmat('../Datasets/radar/Interpolation_orthogonal/Y_test.mat')
Y_test = Y_test['Y_test']
lbl_train = sio.loadmat('../Datasets/radar/Interpolation_orthogonal/lbl_train.mat')
lbl_train = lbl_train['lbl_train']
lbl_test = sio.loadmat('../Datasets/radar/Interpolation_orthogonal/lbl_test.mat')
lbl_test = lbl_test['lbl_test']


# In[3]:


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)


# In[4]:


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
    


# In[5]:


print("X train shape: ", X_train.shape)
print("X test shape: ", X_test.shape)
print("Y train shape: ", Y_train.shape)
print("Y test shape: ", Y_test.shape)
print("Label train shape: ", lbl_train.shape)
print("Label test shape: ", lbl_test.shape)


# In[6]:


np.random.seed(2020)

X_train, Y_train, lbl_train = shuffle(X_train[:], Y_train[:], lbl_train[:], random_state = 2020)
X_test, Y_test, lbl_test = shuffle(X_test[:], Y_test[:], lbl_test[:], random_state = 2020)


# In[7]:


print(Y_train[:5,:])
print(lbl_train[:5,:])


# In[8]:


print(Y_test[:5,:])
print(lbl_test[:5,:])


# In[9]:

def residual_stack(x, f):
    
    if x.shape[1] != f:
        x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    
    # residual unit 1    
    x_shortcut = x
    x = Conv1D(f, 5, strides=1, padding="same", data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 5, strides=1, padding="same", data_format='channels_last')(x)
    x = BatchNormalization()(x)
    
    # add skip connection
    if x.shape[1] == x_shortcut.shape[1]:
        x = Add()([x, x_shortcut])
    else:
        raise Exception('Skip Connection Failure!')
    
    x = Activation('relu')(x)  
    # residual unit 2    
    x_shortcut = x
    x = Conv1D(f, 5, strides=1, padding="same", data_format='channels_last')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(f, 5, strides = 1, padding = "same", data_format='channels_last')(x)
    
    # add skip connection
    if x.shape[1] == x_shortcut.shape[1]:
        x = Add()([x, x_shortcut])
    else:
          raise Exception('Skip Connection Failure!')
    x = Activation('relu')(x)  
    # max pooling layer
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    return x


def RecComModel(input_shape):
    """   
    Arguments:
    input_shape -- shape of the inputs of the dataset
        (height, width, channels) as a tuple.  
        Note that this does not include the 'batch' as a dimension.
        If you have a batch like 'X_train', 
        then you can provide the input_shape using
        X_train.shape[1:]
    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input placeholder as a tensor with shape input_shape.
    X_input = Input(input_shape)

    X = residual_stack(X_input, 32)
    X = residual_stack(X, 32)
    X = residual_stack(X, 32)
    X = residual_stack(X, 32)
    X = residual_stack(X, 32)
    X = residual_stack(X, 32)

    X = Flatten()(X)
    
    X = Dense(128, activation='selu')(X)
    X = AlphaDropout(0.6)(X)
    
    X = Dense(128, activation='selu')(X)
    X = AlphaDropout(0.6)(X)

    X = Dense(23, activation='softmax')(X)
    
    # Create model. This creates your Keras model instance, you'll use this instance to train/test the model.
    model = Model(inputs = X_input, outputs = X)
    model.summary()
    
    return model


# In[38]:


model = RecComModel(X_train.shape[1:])


# In[43]:


output_path = '../Results/Radar_RNN/resnet_k5/'


clr_triangular = CyclicLR(mode='triangular', base_lr=1e-7, max_lr=1e-3, step_size= 4 * (X_train.shape[0] // 1200))

c=[clr_triangular,ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

model.compile(optimizer=optimizers.Adam(1e-7), loss='categorical_crossentropy', metrics=['accuracy'])


# In[44]:


#opt = optimizers.Adam(0.05)

#model.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ["accuracy"])


# In[45]:


output_path = '../Results/Radar_RNN/resnet_k5/'


# In[ ]:


Train = True

if Train:
    #c = [EarlyStopping(monitor='val_loss', patience=20),
    #            ModelCheckpoint(filepath= output_path +'best_model.h5', monitor='val_loss', save_best_only=True)]

    history = model.fit(X_train, Y_train, epochs = 200, batch_size = 1200, callbacks = c, validation_data=(X_test, Y_test))

    with open(output_path +'history_rnn.json', 'w') as f:
        json.dump(history.history, f)
    model_json = model.to_json()
    with open(output_path +'model_rnn.json', "w") as json_file:
        json_file.write(model_json)
else:
    model.load_weights(output_path +'best_model.h5')
    with open(output_path +'history_rnn.json', 'r') as f:
            history = json.load(f)

# In[33]:
model.load_weights(output_path +'best_model.h5')

# In[ ]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['test', 'val'])
plt.show()
plt.savefig(output_path + '\graphs\model_loss.pdf')


# In[ ]:


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


# In[ ]:


def getFontColor(value):
    if np.isnan(value):
        return "black"
    elif value < 0.2:
        return "black"
    else:
        return "white"



# In[ ]:


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
    plt.savefig(output_path + '\graphs\confmat_'+str(snr)+'.pdf')
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
#print(acc)


# In[ ]:


with open(output_path +'acc.json', 'w') as f:
        json.dump(acc, f)
        
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("Classification Accuracy on Radar Dataset")
plt.savefig(output_path + '\graphs\clas_acc.pdf')


# In[ ]:





# In[ ]:





# In[ ]:




