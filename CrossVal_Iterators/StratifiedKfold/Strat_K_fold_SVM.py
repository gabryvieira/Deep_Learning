import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time
import xlsxwriter

#global variables
filename = "../../Datasets/exovar_features.txt"
file_to_write = '../Iterators_Score/ExoVar_Score/StratKfold_SVM_mcc.xlsx'
n_splits = 10
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = 'roc_auc'

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines

    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class

# training
def trainingSample(input_x, input_y):


    clf = svm.SVC(kernel='linear', C=1)
    skf = StratifiedKFold(n_splits=n_splits, random_state=0)
    mkp = make_pipeline(preprocessing.StandardScaler(), clf)
    trainingtime = time.time()
    scores = cross_val_score(mkp, input_x, input_y, cv=skf, n_jobs=-1, scoring=scoring)
    execTime = (time.time() - trainingtime)
    print("Training time:  %s seconds" % execTime)
    print(scores.mean())

    classtime = time.time()
    predicted = cross_val_predict(mkp, input_x, input_y, cv=skf, n_jobs=-1)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predicted)
    return scores.mean()


def save_score(mean, exectime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time')
    ws.write('B2', '%s' % exectime)
    ws.write('A3', 'CV')
    ws.write('B3', '%s' % n_splits)
    wk.close()

if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    mean = trainingSample(x, y)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    save_score(mean, execTime)