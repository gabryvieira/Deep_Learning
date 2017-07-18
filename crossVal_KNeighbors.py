import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
import time

# global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossKNeighbors_score.txt'
num_neighbors = 10 # 5 is default
weight = 'uniform'
cv = 10


def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY):

    neigh = KNeighborsClassifier(n_neighbors=num_neighbors, weights=weight)
    mkp = make_pipeline(preprocessing.StandardScaler(), neigh)
    scores = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    print(scores)
    return scores, scores.mean()


def save_score(scores, mean, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % scores)
    f.write('\nAverage result: %s' % mean)
    f.write('\nExecution time: %s' % execTime)
    f.write('\nNum of neighbors: %s' % num_neighbors)
    f.write('\nWeights: %s' % weight)
    f.write('\nCV: %s' % cv)
    f.close()

if __name__ == '__main__':
    start_time = time.time()
    x, y = read_data(filename)

    score, mean = training_and_testing(x, y)
    # model, x_test, y_test, score = training_and_testing(x, y, testSize)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(score, mean, execTime)