import numpy as np

def load_data():
    return np.loadtxt("./Practica2/p2/data/houses.txt", delimiter=',')

def getNormalizedData():
    data = load_data()
    dataN, mu, sigma = mlr.zscore_normalize_features(data[:, :4])
    # Apend last column of data to dataN
    dataN = np.append(dataN, data[:, 4].reshape(-1, 1), axis=1)
    return [dataN, mu, sigma]