import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time
import xlsxwriter

#global variables
n_estimators = 20
n_splits = 10
filename = "../../Datasets/exovar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/GroupKfold_AdaBoost_mcc.xlsx'
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = mcc_scorer

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    groups = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str')
    file_data_gene = np.ravel(groups[:, 0])
    geneID = [i.split(':', 1)[0] for i in file_data_gene]  # remove : and everything after in array

    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class, geneID

def showScore_CrossValRF(input_x, input_y, groups):

    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
    X_train, X_test, y_train, y_test = train_test_split(input_x, input_y, test_size=0.5)
    training_time = time.time()
    for i in range(1, 10):
        ada.fit(X_train, y_train)
    exec_time = (time.time() - training_time)
    print("Training time:  %s seconds" % exec_time)
    classtime = time.time()
    for a in range(1, 10):
        predictions = ada.predict(X_test)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predictions)


    # kfold = GroupKFold(n_splits=n_splits)
    # mkp = make_pipeline(preprocessing.StandardScaler(), ada)  # scaler e pipeline
    # trainingtime = time.time()
    # #scores = cross_val_score(mkp, input_x, input_y, groups=groups, cv=kfold, n_jobs=-1, scoring=scoring)
    #
    # execTime = (time.time() - trainingtime)
    # print("Training time:  %s seconds" % execTime)
    # print(scores.mean())
    #
    # classtime = time.time()
    # #predicted = cross_val_predict(ada, input_x, input_y, groups=groups, cv=kfold)
    # execTime = (time.time() - classtime)
    # print("Classification time:  %s seconds" % execTime)
    # print(predicted)

    return scores.mean()

def save_score(mean, execTime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score:')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time:')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'CV:')
    ws.write('B3', '%s' % n_splits)
    ws.write('A4', 'Number of estimators:')
    ws.write('B4', '%s' % n_estimators)
    wk.close()


if __name__ == '__main__':

    start_time = time.time()
    x, y, groups = read_data(filename)


    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    mean = showScore_CrossValRF(x, y, groups)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(mean, execTime)