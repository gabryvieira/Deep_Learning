from sklearn import tree
import numpy as np
from sklearn.model_selection import cross_val_score
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline

#global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossDecTree_score.txt'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY):

    dt = DecisionTreeClassifier(random_state=0)
    mkp = make_pipeline(preprocessing.StandardScaler(), dt)
    scores = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    print(scores)
    return scores, scores.mean()


def save_score(scores, mean, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % scores)
    f.write('\nAverage result: %s' % mean)
    f.write('\nExecution time: %s' % execTime)
    f.write('\nCV: %s' % cv)
    f.close()


if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    scores, mean = training_and_testing(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(scores, mean, execTime)