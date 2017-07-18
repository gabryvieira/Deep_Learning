import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model, metrics
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

#global variables
filename = "../../Datasets/exovar_features.txt"
file_to_write = '../Iterators_Score/StratKfold_LinearReg.txt'



def read_data(file):
    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (
    len(true_class), len(file_data[0, :-1]), positive_fraction))

    # in linear regression we can use only one feature, cause the x and y must have the same size
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY):

    mcc_scorer = make_scorer(matthews_corrcoef)

    totalScore = 0
    model = linear_model.LinearRegression()
    skf = StratifiedKFold(n_splits=5, random_state=0)


    for train_index, test_index in skf.split(inputX, inputY):
        x_train, x_test = inputX[train_index], inputX[test_index]  # sacar valores dos respectivos indices
        y_train, y_test = inputY[train_index], inputY[test_index]
        model.fit(x_train, y_train)
        mkp = make_pipeline(preprocessing.StandardScaler(), model)
        scores = cross_val_score(mkp, inputX, inputY, cv=skf, n_jobs=-1)
        print(scores)
        totalScore += scores.mean()

    average = totalScore / skf.get_n_splits()
    return average



def save_score(mean, execTime):
    f = open(file_to_write, 'w')
    f.write('Average score: %s ' % mean)
    f.write('\nExecution time: %s seconds' % execTime)


if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    training_and_testing(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    #save_score(mean, execTime)

