import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import time
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

# global variables
filename = "Datasets/exovar_features.txt"
file_to_write = 'ScoreFiles/crossLinearReg_score.txt'



def read_data(file):
    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 0])
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (
    len(true_class), len(file_data[0, :-1]), positive_fraction))

    # in linear regression we can use only one feature, cause the x and y must have the same size
    return file_data[:, 1:], true_class


def training_and_testing(inputX, inputY):

    mcc_scorer = make_scorer(matthews_corrcoef)
    model = linear_model.LinearRegression()
    scores = cross_val_score(model, inputX, inputY, cv=5, n_jobs=-1)
    #mkp = make_pipeline(preprocessing.StandardScaler(), model)
    #scores = cross_val_score(mkp, inputX, inputY, cv=5, n_jobs=-1)
    print(scores.mean())
    return scores, scores.mean()



def save_score(scores, mean, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % scores)
    f.write('\nMean:  %s' % mean)
    f.write('\nExecution time: %s' % execTime)


if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    scores, mean = training_and_testing(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(scores, mean, execTime)

