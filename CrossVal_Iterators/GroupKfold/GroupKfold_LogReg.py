import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
import time
import xlsxwriter

#global variables
from sklearn.preprocessing import LabelEncoder

filename = "../../Datasets/humvar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/GroupKfold_LogReg_roc_auc.xlsx'
n_splits = 10
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = mcc_scorer

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])
    groups = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str')
    file_data_gene = np.ravel(groups[:, 0])
    geneID = [i.split(':', 1)[0] for i in file_data_gene]  # remove : and everything after in array

    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (
    len(true_class), len(file_data[0, :-1]), positive_fraction))

    # in linear regression we can use only one feature, cause the x and y must have the same size
    return file_data[:, 2:], true_class, geneID

def cross_val(inputX, inputY, groups):

    lr = linear_model.LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.5)
    training_time = time.time()
    for i in range(1, 10):
        lr.fit(X_train, y_train)
    exec_time = (time.time() - training_time)
    print("Training time:  %s seconds" % exec_time)
    classtime = time.time()
    for a in range(1, 10):
        predictions = lr.predict(X_test)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predictions)
    # kfold = GroupKFold(n_splits=n_splits)
    # mkp = make_pipeline(preprocessing.StandardScaler(), lr)
    # trainingtime = time.time()
    # scores = cross_val_score(mkp, inputX, inputY, cv=kfold, groups=groups, n_jobs=-1, scoring=scoring)
    # execTime = (time.time() - trainingtime)
    # print("Training time:  %s seconds" % execTime)
    # print(scores.mean())
    #
    # classtime = time.time()
    # predicted = cross_val_predict(mkp, inputX, inputY, cv=kfold, groups=groups, n_jobs=-1)
    # execTime = (time.time() - classtime)
    # print("Classification time:  %s seconds" % execTime)
    # print(predicted)
    return scores.mean()


def save_score(mean, execTime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'CV')
    ws.write('B3', '%s' % n_splits)
    wk.close()



if __name__ == '__main__':

    start_time = time.time()
    x, y, groups = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    mean = cross_val(x, y, groups)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(mean, execTime)

