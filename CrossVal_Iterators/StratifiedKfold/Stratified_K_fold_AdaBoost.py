import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time


#global variables
filename = "../../Datasets/humvar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/StratKfold_AdaBoost.txt'
n_splits = 10
n_estimators = 100

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines

    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class

# training
def trainingSample(input_x, input_y):

    mcc_scorer = make_scorer(matthews_corrcoef)

    ada = AdaBoostClassifier(n_estimators=n_estimators, random_state=0)
    skf = StratifiedKFold(n_splits=n_splits, random_state=0)
    mkp = make_pipeline(preprocessing.StandardScaler(), ada)
    scores = cross_val_score(mkp, input_x, input_y, cv=skf, n_jobs=-1)
    print(scores.mean())
    return scores.mean()


def save_score(mean, exectime):
    f = open(file_to_write, 'w')
    #str_score = ''.join(str(score))
    f.write('Average score: %s' % mean)
    f.write('\nExecution time: %s seconds' % exectime)
    f.write('\nCV: %s' % n_splits)
    f.write('\nNumber of estimators: %s' % n_estimators)
    f.close()

if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    mean = trainingSample(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(mean, execTime)