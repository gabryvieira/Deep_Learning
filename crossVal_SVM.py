import numpy as np
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time


#global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossSVM_score.txt'
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
    #scores = cross_val_score(clf, x, y, cv=cv, n_jobs=-1)
    mkp = make_pipeline(preprocessing.StandardScaler(), clf)
    scores = cross_val_score(mkp, input_x, input_y, cv=cv, n_jobs=-1)
    return scores, scores.mean()


def save_score(score, mean, exectime):
    f = open(file_to_write, 'w')
    #str_score = ''.join(str(score))
    f.write('Score: %s' % score)
    f.write('\nAverage: %s' % mean)
    f.write('\nExecution time: %s' % exectime)
    f.write('\nCV: %s' % cv)
    f.close()

if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    scores, mean = trainingSample(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(scores, mean, execTime)