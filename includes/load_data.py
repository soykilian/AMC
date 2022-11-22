import h5py
import sklearn
import scipy.io as io
import numpy as np

def load_data(dataset_path : str, subset : str,AF : bool):
    with h5py.File(dataset_path +'X_'+ subset +'.mat', 'r') as f:
        X = np.array(f['X_' + subset]).T
    Y = io.loadmat(dataset_path + 'Y_'+subset+'.mat')
    Y = Y['Y_' + subset]

    if AF:
        I_tr = X[:, :, 0]
        Q_tr = X[:, :, 1]
        X_tr = I_tr + 1j * Q_tr

        X[:, :, 1] = np.arctan2(Q_tr, I_tr) / np.pi
        X[:, :, 0] = np.abs(X_tr)

        del I_tr
        del Q_tr
        del X_tr
    np.random.seed(2022)
    X, Y = sklearn.utils.shuffle(X[:], Y[:], random_state=2022)
    return X, Y

