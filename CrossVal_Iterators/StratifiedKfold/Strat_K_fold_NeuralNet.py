import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, StratifiedKFold
import time
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
import xlsxwriter
#global variables
filename = "../../Datasets/humvar_features.txt"
file_to_write = '../Iterators_Score/HumVar_Score/StratKfold_NeuralNet_mcc.xlsx'
n_splits = 10
mcc_scorer = make_scorer(matthews_corrcoef)
scoring = 'accuracy'
#n_hidden_layers = 6
#neurons_per_layer = 100

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    #print(true_class)
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY, num_layers, num_neurons_layer):



    mlp = MLPClassifier(hidden_layer_sizes=(num_neurons_layer, num_layers), solver='adam', alpha=1e-5, random_state=0)
    skf = StratifiedKFold(n_splits=n_splits, random_state=0)

    # ciclo pode ser usado apenas para verificar que indices dos dados estao a ser treinados e testados
    # for train_index, test_index in skf.split(inputX, inputY):
    #     x_train, x_test = inputX[train_index], inputX[test_index] # sacar valores dos respectivos indices
    #     y_train, y_test = inputY[train_index], inputY[test_index]
    #     mlp.fit(x_train, y_train)
    #     scores = mlp.score(x_test, y_test)
    #     scores = cross_val_score(mkp, x_test, y_test, cv=skf, n_jobs=-1)

    mkp = make_pipeline(preprocessing.StandardScaler(), mlp)
    trainingtime = time.time()
    scores = cross_val_score(mkp, inputX, inputY, cv=skf, n_jobs=-1, scoring=scoring)
    execTime = (time.time() - trainingtime)
    print("Training time:  %s seconds" % execTime)
    print(scores.mean())

    classtime = time.time()
    predicted = cross_val_predict(mkp, inputX, inputY, cv=skf, n_jobs=-1)
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
    x, y = read_data(filename)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    training_time = time.time()
    mean = training_and_testing(x, y, n_hidden_layers, neurons_per_layer)

    execTime = (time.time() - training_time)
    print("Training time:  %s seconds" % execTime)
    #
    #save_score(mean, n_hidden_layers, neurons_per_layer, execTime)

