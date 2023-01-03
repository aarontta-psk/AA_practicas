import sklearn.svm as skl_svm
import sklearn.metrics as skm

def calc_gamma(sigma):
    return 1 / (2 * sigma**2)

def svm(X_train, y_train, X_val, y_val, kernel_type, c, sigma):
    # gamma value ignored on linear kernel
    svm = skl_svm.SVC(kernel=kernel_type, C=c, gamma=calc_gamma(sigma))
    svm.fit(X_train, y_train.ravel())
    
    if X_val is not None and y_val is not None:
        predict = svm.predict(X_val)
        accuracy = skm.accuracy_score(y_val, predict)

        return svm, predict, accuracy
    else:
        return svm