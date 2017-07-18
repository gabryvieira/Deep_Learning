import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
import time

# global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossLogReg_score.txt'
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
    #without pipeline
    #scores = cross_val_score(lr, inputX, inputY, cv=5, n_jobs=-1)

    #with pipeline
    mkp = make_pipeline(preprocessing.StandardScaler(), lr)
    scores = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    print(scores.mean())
    return scores, scores.mean()


def save_score(score, mean, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % score)
    f.write('\nMean:  %s' % mean)
    f.write('\nExecution time: %s' % execTime)
    f.write('\nCV: %s' % cv)
    f.close()



if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    score, mean = cross_val(x, y)
    # model, x_test, y_test, score = training_and_testing(x, y, testSize)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(score, mean, execTime)

