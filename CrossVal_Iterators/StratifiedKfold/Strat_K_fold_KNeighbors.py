import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer, matthews_corrcoef
import time
import xlsxwriter

# global variables
filename = "../../Datasets/humvar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/StratKfold_KNeighbors_roc_auc.xlsx'
num_neighbors = 10 # 5 is default
weight = 'uniform'
n_splits = 5
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = mcc_scorer


def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY):
    skf = StratifiedKFold(n_splits=n_splits, random_state=0)
    neigh = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight)
    mkp = make_pipeline(preprocessing.StandardScaler(), neigh)
    trainingtime = time.time()
    scores = cross_val_score(mkp, inputX, inputY, cv=skf, n_jobs=-1, scoring=scoring)
    execTime = (time.time() - trainingtime)
    print("Training time:  %s seconds" % execTime)
    print(scores.mean())

    classtime = time.time()
    predicted = cross_val_predict(mkp, inputX, inputY, cv=skf)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predicted)

    return scores, scores.mean()


def save_score(scores, mean, execTime):

    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score:')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time:')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'CV')
    ws.write('B3', '%s' % n_splits)
    ws.write('A4', 'Num of neighbors:')
    ws.write('B4', '%s' % num_neighbors)
    ws.write('A5', 'Weight:')
    ws.write('B5', '%s' % weight)
    wk.close()


if __name__ == '__main__':
    start_time = time.time()
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    score, mean = training_and_testing(x, y)

    execTime = (time.time() - training_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(score, mean, execTime)