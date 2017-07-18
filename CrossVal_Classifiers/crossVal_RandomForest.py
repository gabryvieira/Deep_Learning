import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import time
import xlsxwriter

#global variables
filename = "../Datasets/humvar_features.txt"
file_to_write = '../ScoreFiles/HumVar_crossRandomForest_score.xlsx'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class)*100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class

def showScore_CrossValRF(input_x, input_y):
    num_trees = 10
    #kfold = cross_validation.KFold(n=num_instances, n_folds=num_folds, shuffle=False, random_state=seed)
    model = RandomForestClassifier(max_depth=None, n_estimators=num_trees, max_features=None, random_state=0)
    model.fit(input_x, input_y)
    print("Predictions:")
    print(model.predict(input_x))
    scores = matthews_corrcoef(input_y, model.predict(input_x))
    print(scores)
    #mkp = make_pipeline(preprocessing.StandardScaler(), model)
    #scores = cross_val_score(mkp, input_x, input_y, cv=cv, n_jobs=-1)
    return scores, scores.mean(), num_trees

def save_score(scores, mean, trees, execTime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score:')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time:')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'Number of trees:')
    ws.write('B3', '%s' % trees)
    ws.write('A4', 'CV:')
    ws.write('B4', '%s' % cv)
    wk.close()


if __name__ == '__main__':

    start_time = time.time()
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    score, mean, trees = showScore_CrossValRF(x, y)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)

    #save_score(score, mean, trees, execTime)