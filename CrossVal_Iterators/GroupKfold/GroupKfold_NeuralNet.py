import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, GroupKFold
import time

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import xlsxwriter

#global variables
filename = "../../Datasets/humvar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/GroupKfold_NeuralNet_roc_auc.xlsx'
n_splits = 10
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = 'accuracy'

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    groups = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str')
    file_data_gene = np.ravel(groups[:, 0])
    geneID = [i.split(':', 1)[0] for i in file_data_gene] # remove : and everything after in array


    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class, geneID


def training_and_testing(inputX, inputY, num_layers, num_neurons_layer, groups):

    mcc_scorer = make_scorer(matthews_corrcoef)
    #training
    totalScore = 0
    mlp = MLPClassifier(hidden_layer_sizes=(num_neurons_layer, num_layers), solver='adam', alpha=1e-5, random_state=0)
    kfold = GroupKFold(n_splits=n_splits)
    # X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.5)
    # training_time = time.time()
    # for i in range(1, 10):
    #     mlp.fit(X_train, y_train)
    # exec_time = (time.time() - training_time)
    # print("Training time:  %s seconds" % exec_time)
    # classtime = time.time()
    # for a in range(1, 10):
    #     predictions = mlp.predict(X_test)
    # execTime = (time.time() - classtime)
    # print("Classification time:  %s seconds" % execTime)
    # print(predictions)
    #
    # for train_index, test_index in kfold.split(inputX, inputY, groups):
    #     x_train, x_test = inputX[train_index], inputX[test_index] # sacar valores dos respectivos indices
    #     y_train, y_test = inputY[train_index], inputY[test_index]
    #     #print(train_index, test_index)
    #     #mlp.fit(x_train, y_train)
    mkp = make_pipeline(preprocessing.StandardScaler(), mlp)
    trainingtime = time.time()
    scores = cross_val_score(mkp, inputX, inputY, cv=kfold, n_jobs=-1, groups=groups, scoring=scoring)
    execTime = (time.time() - trainingtime)
    print("Training time:  %s seconds" % execTime)
    print(scores.mean())

    classtime = time.time()
    predicted = cross_val_predict(mkp, inputX, inputY, groups=groups, cv=kfold, n_jobs=-1)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predicted)
    return scores.mean()

def save_score(mean, n_hidden_layers, neurons_per_layer, execTime):
    wk = xlsxwriter.Workbook(file_to_write)
    ws = wk.add_worksheet()
    ws.set_column('A:A', 20)
    ws.write('A1', 'Average score:')
    ws.write('B1', '%s' % mean)
    ws.write('A2', 'Execution time:')
    ws.write('B2', '%s' % execTime)
    ws.write('A3', 'Num of hidden layers:')
    ws.write('B3', '%s' % n_hidden_layers)
    ws.write('A4', 'Neurons per layer:')
    ws.write('B4', '%s' % neurons_per_layer)
    ws.write('A5', 'CV:')
    ws.write('B5', '%s' % n_splits)
    wk.close()



if __name__ == '__main__':

    n_hidden_layers = int(input("Number of hidden layers: "))
    neurons_per_layer = int(input("Neurons per layer: "))

    start_time = time.time()
    x, y, groups = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    mean = training_and_testing(x, y, n_hidden_layers, neurons_per_layer, groups)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)
    #
    save_score(mean, n_hidden_layers, neurons_per_layer, execTime)

