import h5py
import numpy as np
import scipy.io as io
path = '/home/maria/'
dataset_path = path + 'Dataset_trials/'
with h5py.File(dataset_path +'X_train.mat', 'r') as f:
    X_train = np.array(f['X_train']).T
print(X_train.shape)
with h5py.File(dataset_path +'X_test.mat', 'r') as f:
    X_test = np.array(f['X_test']).T
print(X_test.shape)
lbl_train = io.loadmat(dataset_path + 'lbl_train.mat')['lbl_train']
lbl_test = io.loadmat(dataset_path + 'lbl_test.mat')['lbl_test']
print(lbl_test.shape)
print(lbl_train.shape)
Y_train = io.loadmat(dataset_path + 'Y_train.mat')
Y_train = Y_train['Y_train']
Y_test = io.loadmat(dataset_path + 'Y_test.mat')
Y_test = Y_test['Y_test']
print(Y_train.shape)
print(Y_test.shape)
acc={}
snrs = [-12,-10,-8,-6,-4,-2,0,2,4,6,8,10,12,14,16,18,20]
for snr in snrs:
    test_SNRs = list(map(lambda x: lbl_test[x][1], range(0,X_test.shape[0])))
    test_X_i = X_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
    print(snr)
    if (test_X_i.size == 0):
        print("TRAZA")
    print("------------------------------------------------------------------")
    test_Y_i = Y_test[[i for i,x in enumerate(test_SNRs) if x==snr]]
