import numpy as np
from sklearn.model_selection import cross_val_score
import time
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer

#global variables
filename = "Datasets/humvar_features.txt"
file_to_write = 'ScoreFiles/HumVar_crossNeuralNet_score.txt'
cv = 10

def read_data(file):

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
    true_class = np.ravel(file_data[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)

    print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
    return file_data[:, 2:], true_class


def training_and_testing(inputX, inputY, num_layers, num_neurons_layer):

    mcc_scorer = make_scorer(matthews_corrcoef)
    #training

    mlp = MLPClassifier(hidden_layer_sizes=(num_neurons_layer, num_layers), solver='adam', alpha=1e-5, random_state=0)
    mkp = make_pipeline(preprocessing.StandardScaler(), mlp)
    scores = cross_val_score(mkp, inputX, inputY, cv=cv, n_jobs=-1)
    print(scores)

    return scores, scores.mean()

def save_score(scores, mean, n_hidden_layers, neurons_per_layer, execTime):
    f = open(file_to_write, 'w')
    f.write('Score: %s ' % scores)
    f.write('\nAverage result: %s' % mean)
    f.write('\nNumber of hidden layers: %s' % n_hidden_layers)
    f.write('\nNumber of neurons per layer: %s' % neurons_per_layer)
    f.write('\nExecution time: %s' % execTime)
    f.write('\nCV: %s' % cv)
    f.close()



if __name__ == '__main__':

    n_hidden_layers = int(input("Number of hidden layers: "))
    neurons_per_layer = int(input("Neurons per layer: "))

    start_time = time.time()
    x, y = read_data(filename)

    scores, mean = training_and_testing(x, y, n_hidden_layers, neurons_per_layer)

    execTime = (time.time() - start_time)
    print("Execution time:  %s seconds" % execTime)

    save_score(scores, mean, n_hidden_layers, neurons_per_layer, execTime)

