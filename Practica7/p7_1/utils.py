import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def obtain_data(data_path):
    data = sio.loadmat(data_path)
    # print(data.keys()) # tells .mat keys
    if data_path == './data/ex6data3.mat':
        return data['X'], data['y'], data['Xval'], data['yval']
    else:
        return data['X'], data['y']

def plot_svm(res_path, title, X, y, svm):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    x1, x2 = np.meshgrid(x1, x2)
    y_predict = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    plt.figure()
    plt.title(title)

    plt.scatter(X[(y==1).ravel(), 0], X[(y==1).ravel(), 1], color='blue', marker='+')
    plt.scatter(X[(y==0).ravel(), 0], X[(y==0).ravel(), 1], color='red', marker='o')
    plt.contour(x1, x2, y_predict)

    plt.savefig(res_path)
    plt.close('all')