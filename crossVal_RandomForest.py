import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time


#global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossRandomForest_score.txt'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class

def showScore_CrossValRF(input_x, input_y):
    num_trees = 10
    mcc_scorer = make_scorer(matthews_corrcoef)
    #kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, shuffle=False, random_state=seed)
    model = RandomForestClassifier(max_depth=None, n_estimators=num_trees, max_features=None, random_state=0)
    mkp = make_pipeline(preprocessing.StandardScaler(), model)
    scores = cross_val_score(mkp, input_x, input_y, cv=cv, n_jobs=-1)
    print(scores.mean())
    return scores, scores.mean(), num_trees

def save_score(scores, mean, trees, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % scores)
    f.write('\nAverage result: %s' % mean)
    f.write('\nNumber of trees: %s' % trees)
    f.write('\nExecution time: %s' % execTime)
    f.write('\nCV: %s' % cv)
    f.close()


if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)


    score, mean, trees = showScore_CrossValRF(x, y)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(score, mean, trees, execTime)