import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import time
import xlsxwriter

# global variables
filename = "../Datasets/exovar_features.txt"
file_to_write = '../ScoreFiles/crossLogReg_score.xlsx'
cv = 10


def read_data(file):
    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (
    len(true_class), len(file_data[0, :-1]), positive_fraction))

    # in linear regression we can use only one feature, cause the x and y must have the same size
    return file_data[:, 2:], true_class

def cross_val(inputX, inputY):
    mcc_scorer = make_scorer(matthews_corrcoef)
    lr = linear_model.LogisticRegression()
    lr.fit(inputX, inputY)
    # print("Predictions:")
    # print(lr.predict(inputX))
    print(lr.predict_proba(inputX)[:,1])
    scores = roc_auc_score(inputY, lr.predict_proba(inputX)[:,1])
    print(scores)
    #without pipeline
    #scores = cross_val_score(lr, inputX, inputY, cv=5, n_jobs=-1)

    #with pipeline
    #mkp = make_pipeline(preprocessing.StandardScaler(), lr)
    #scores = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    return scores, scores.mean()


def save_score(score, mean, execTime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'CV')
    ws.write('B3', '%s' % cv)
    wk.close()



if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    score, mean = cross_val(x, y)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(score, mean, execTime)

