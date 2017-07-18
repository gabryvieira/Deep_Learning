from sklearn import tree
import numpy as np
from sklearn.metrics import matthews_corrcoef, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import time

from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import xlsxwriter

#global variables
filename = "../Datasets/humvar_features.txt"
file_to_write = '../ScoreFiles/HumVar_crossDecTree_score.xlsx'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY):

    dt = DecisionTreeClassifier(random_state=0, min_samples_leaf=5)
    dt.fit(inputX, inputY)
    print("Predictions:")
    print(dt.predict(inputX))
    score = accuracy_score(inputY, dt.predict(inputX))
    #score = roc_auc_score(inputY, dt.predict_proba(inputX)[:,1])
    print(score)
    #mkp = make_pipeline(preprocessing.StandardScaler(), dt)
    #score = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    #print(scores)
    return score, score.mean()


def save_score(scores, mean, execTime):
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

    # encoder = LabelEncoder()
    # encoder.fit(y)
    # y = encoder.transform(y)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    scores, mean = training_and_testing(x, y)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(scores, mean, execTime)