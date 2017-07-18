import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time
import xlsxwriter

#global variables
filename = "../Datasets/humvar_features.txt"
file_to_write = '../ScoreFiles/HumVar_crossSVM_score.xlsx'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class

# training
def trainingSample(input_x, input_y):

    mcc_scorer = make_scorer(matthews_corrcoef)
    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(input_x, input_y)
    print("Predictions:")
    print(clf.predict(input_x))
    scores = accuracy_score(input_y, clf.predict(input_x))
    print(scores)
    #scores = cross_val_score(clf, x, y, cv=cv, n_jobs=-1)
    #mkp = make_pipeline(preprocessing.StandardScaler(), clf)
    #scores = cross_val_score(mkp, input_x, input_y, cv=cv, n_jobs=-1)
    return scores, scores.mean()


def save_score(score, mean, exectime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    # Widen the first column to make the text clearer.
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time')
    ws.write('B2', '%s' % exectime)
    ws.write('A3', 'CV')
    ws.write('B3', '%s' % cv)
    wk.close()

if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    scores, mean = trainingSample(x, y)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(scores, mean, execTime)