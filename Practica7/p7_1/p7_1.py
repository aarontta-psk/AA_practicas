import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import sklearn.svm as skl_svm
import sklearn.metrics as skm

def obtain_data(data_path):
    data = sio.loadmat(data_path)
    # print(data.keys())
    if data_path == 'data/ex6data3.mat':
        return data['X'], data['y'], data['Xval'], data['yval']
    else:
        return data['X'], data['y']

def plot_graph(res_path, svm, X, y):
    x1 = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
    x2 = np.linspace(X[:, 1].min(), X[:, 1].max(), 100)

    x1, x2 = np.meshgrid(x1, x2)
    y_predict = svm.predict(np.array([x1.ravel(), x2.ravel()]).T).reshape(x1.shape)

    plt.figure()

    plt.scatter(X[(y==1).ravel(), 0], X[(y==1).ravel(), 1], color='blue', marker='+')
    plt.scatter(X[(y==0).ravel(), 0], X[(y==0).ravel(), 1], color='red', marker='o')
    plt.contour(x1, x2, y_predict)

    plt.savefig(res_path)
    plt.close('all')

def try_svm(data_path, res_path, kernel_type, c, sigma):
    X, y = obtain_data(data_path)

    gamma_value = 1 / (2 * sigma**2)
    svm = skl_svm.SVC(kernel=kernel_type, C=c, gamma=gamma_value)
    svm.fit(X, y.ravel())

    plot_graph(res_path, svm, X, y)

def apply_svm(data_path, kernel_type, c, sigma):
    X, y, X_val, y_val = obtain_data(data_path)

    gamma_value = 1 / (2 * sigma**2)
    svm = skl_svm.SVC(kernel=kernel_type, C=c, gamma=gamma_value)
    svm.fit(X, y.ravel())
    predict = svm.predict(X_val)

    print('Accuracy: ', skm.accuracy_score(y_val, predict))

    return skm.accuracy_score(y_val, predict), svm, X, y

def get_best_svm_parameters():
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    svm, X, y = skl_svm.SVC(), np.zeros(1), np.zeros(1)

    best_acc, c_id, sigma_id = -1, 0, 0
    for c in range(len(values)):
        for sigma in range(len(values)):
            accuracy, new_svm, new_X, new_y = apply_svm('data/ex6data3.mat', 'rbf', values[c], values[sigma])
            
            if(best_acc < accuracy):
                best_acc, c_id, sigma_id = accuracy, c, sigma
                svm, X, y = new_svm, new_X, new_y
                
    print("C: ", values[c_id])
    print("Sigma: ", values[sigma_id])
    plot_graph('results/graph_c_' + str(values[c_id]) + '_sigma_' + str(values[sigma_id]) + '.png', svm, X, y)

def main():
    # comparing c values
    try_svm('data/ex6data1.mat', 'results/graph_c_1.png', 'linear', 1.0, 1.0)
    try_svm('data/ex6data1.mat', 'results/graph_c_2.png', 'linear', 10.0, 1.0)

    # using gaussian kernel
    try_svm('data/ex6data2.mat', 'results/graph_sigma.png', 'rbf', 1.0, 0.1)

    # pick best parameters
    get_best_svm_parameters()

if __name__ == '__main__':
    main()