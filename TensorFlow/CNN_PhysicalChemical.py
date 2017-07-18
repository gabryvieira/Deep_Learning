import time
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Convolution2D, MaxPooling2D, Dropout, Flatten, Convolution1D, MaxPooling1D, \
    Embedding
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
import keras.losses as l
import keras.optimizers as opt
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn import datasets, preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import LabelEncoder
import numpy as np
from keras import backend as K

K.set_image_dim_ordering('tf')


### Script aparte para correr a estrategia physical-chemical properties na CNN

mcc_scorer = make_scorer(matthews_corrcoef)
scoring = mcc_scorer
epochs = 15
batch_size = 32
np_splits = 10
kernel_size = 3
num_strides = 1
filters = 32

properties = ['hydropathy', 'polarity', 'averageMass', 'isoElectricPt', 'vanDerWaals', 'netCharge', 'pK1', 'pK2', 'pKa']

aa_properties = {'hydropathy': {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3},
                 'polarity': {'A': 8.1, 'C': 5.5, 'D': 13, 'E': 10.5, 'F': 5.2, 'G': 9, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9, 'M': 5.7, 'N': 11.6, 'P': 8, 'Q': 12.3, 'R': 10.5, 'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2},
                 'averageMass': {'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2, 'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.1, 'R': 174.2, 'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2},
                 'isoElectricPt': {'A': 6.01, 'C': 5.05, 'D': 2.85, 'E': 3.15, 'F': 5.49, 'G': 6.06, 'H': 7.6, 'I': 6.05, 'K': 9.6, 'L': 6.01, 'M': 5.74, 'N': 5.41, 'P': 6.3, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.6, 'V': 6, 'W': 5.89, 'Y': 5.64},
                 'vanDerWaals': {'A': 67, 'C': 86, 'D': 91, 'E': 109, 'F': 135, 'G': 48, 'H': 118, 'I': 124, 'K': 135, 'L': 124, 'M': 124, 'N': 96, 'P': 90, 'Q': 114, 'R': 148, 'S': 73, 'T': 93, 'V': 105, 'W': 163, 'Y': 141},
                 'netCharge': {'A': 0.007, 'C': -0.037, 'D': -0.024, 'E': 0.007, 'F': 0.038, 'G': 0.179, 'H': -0.011, 'I': 0.022, 'K': 0.018, 'L': 0.052, 'M': 0.003, 'N': 0.005, 'P': 0.24, 'Q': 0.049, 'R': 0.044, 'S': 0.005, 'T': 0.003, 'V': 0.057, 'W': 0.038, 'Y': 0.024},
                 'pK1': {'A': 2.35, 'C': 1.92, 'D': 1.99, 'E': 2.1, 'F': 2.2, 'G': 2.35, 'H': 1.8, 'I': 2.32, 'K': 2.16, 'L': 2.33, 'M': 2.13, 'N': 2.14, 'P': 1.95, 'Q': 2.17, 'R': 1.82, 'S': 2.19, 'T': 2.09, 'V': 2.39, 'W': 2.46, 'Y': 2.2},
                 'pK2': {'A': 9.87, 'C': 10.7, 'D': 9.9, 'E': 9.47, 'F': 9.31, 'G': 9.78, 'H': 9.33, 'I': 9.76, 'K': 9.06, 'L': 9.74, 'M': 9.28, 'N': 8.72, 'P': 10.64, 'Q': 9.13, 'R': 8.99, 'S': 9.21, 'T': 0.0, 'V': 9.74, 'W': 9.41, 'Y': 9.21},
                 'pKa': {'A': 0.0, 'C': 8.18, 'D': 3.9, 'E': 4.07, 'F': 0.0, 'G': 0.0, 'H': 6.04, 'I': 0.0, 'K': 10.54, 'L': 0.0, 'M': 0.0, 'N': 5.41, 'P': 0.0, 'Q': 0.0, 'R': 12.48, 'S': 5.68, 'T': 5.53, 'V': 0.0, 'W': 5.885, 'Y': 10.46}}

# contagem do tempo de pre processamento
start_time = time.time()

codon_aa_map = dict()
for line in open('gct.txt', 'rt'):
    aa, _, _, codons = line.strip('\n').split('\t')
    for codon in codons.split('|'):
            codon_aa_map[codon] = aa


codon_aa_map['XXX'] = '0'
print(codon_aa_map)
for p in properties:
    aa_properties[p]['0']=100   # pode experimentar-se com outro valor, p.ex ‘-100'

for p in properties:
    aa_properties[p]['*']=100   # pode experimentar-se com outro valor, p.ex ‘-100'


filename = '../Datasets/exovar_codon_sequence.txt'


y = np.ravel(np.genfromtxt(filename, delimiter=',', skip_header=0, usecols=1))
y[y==-1]=0 # os -1 colocados a 0

file_data = np.genfromtxt(filename, delimiter=',', skip_header=0, dtype='S')
groups = file_data[:, 0] # geneID



N = len(y)
print(N)
M = len(properties)  # numero de propriedades (= tamanho do vetor que representa cada AA)
print(M)
pos = 30  # posicao do codao com mutacao, nos ficheiros de dados
W = 3  # janela de codoes / AAs a usar


#sequencia original
X1 = np.array([np.array([[aa_properties[p][codon_aa_map[codon]] for p in properties]
                         for codon in x.decode().split('.')[pos-W:pos+W+1]])
                            for x in file_data[:, 2]])#.reshape(N, M*(2*W+1))

# sequencia mutada
X2 = np.array([np.array([[aa_properties[p][codon_aa_map[codon]] for p in properties]
                         for codon in x.decode().split('.')[pos-W:pos+W+1]])
                            for x in file_data[:, 3]])#.reshape(N, M*(2*W+1))

# diferenca
X = X1 - X2
print(X.shape)

execTime = (time.time() - start_time)
print("Pre-processing time: %s seconds" % execTime)


#def baseLineModel():
model = Sequential()
model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(7, M), padding='same', activation='relu', strides=1))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, kernel_initializer='normal', activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    #return model

# sem cross validation (usado apenas para a medicao de tempos de treino e classificacao)#
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
training_time = time.time()
for i in range(1, 10):
    model.fit(X_train, y_train)
exec_time = (time.time() - training_time)
print("Training time:  %s seconds" % exec_time)
classtime = time.time()
for a in range(1, 10):
    predictions = model.predict(X_test)
execTime = (time.time() - classtime)
print("Classification time:  %s seconds" % execTime)


# com cross validation

# #mkp = make_pipeline(preprocessing.StandardScaler(), mlp)
# #scores = cross_val_score(mkp, X, y, cv=kfold, n_jobs=-1, scoring='roc_auc')
# estimators = [('cnn', KerasClassifier(build_fn=baseLineModel, epochs=epochs, batch_size=batch_size,verbose=0))]
# kfold = GroupKFold(n_splits=np_splits)
# #kfold = StratifiedKFold(n_splits=np_splits, random_state=0)
# training_time = time.time()
# pipeline = Pipeline(estimators)
# scores = cross_val_score(pipeline, X, y, cv=kfold, n_jobs=-1, groups=groups, scoring=scoring, verbose=0)
# classification_time = (time.time() - training_time)
# print("Training time:  %s seconds" % classification_time)
# print("Accuracy: %.2f%% (%.2f%%)" % (scores.mean() * 100, scores.std() * 100))




















# #read data
# file = "../Datasets/exovar_features.txt"
# n_splits = 10
#
# file_data = np.genfromtxt(file, delimiter=',', skip_header=0)
# true_class = np.ravel(file_data[:, 1])  # get all lines
# positive_fraction = (sum(true_class) * 100.0) / len(true_class)
# print("Done! %i data points were read, with %i features (%.2f%% positive)" %(len(true_class), len(file_data[0, :-1]), positive_fraction))
# x, y = file_data[:, 2:], true_class
#
# #encoder = LabelEncoder()  # encode class values as integers
# #encoder.fit(y)
# #encoded_y = encoder.transform(y)
# #dummy_y = np_utils.to_categorical(encoded_y) # convert integers to dummy variables
#
#
# encoder = LabelEncoder()  # encode class values as integers
# encoder.fit(y)
# encoded_y = encoder.transform(y)
# x = x.reshape(x.shape + (1,))
# #encoded_y = encoded_y.reshape(encoded_y.shape + (1,))
# #encoded_y = np_utils.to_categorical(encoded_y, 2) # convert integers to dummy variables
# def baselineModel():
#     model = Sequential()
#     model.add(Conv1D(filters=32, kernel_size=3, input_shape=(x.shape[1], 1), padding='same', activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(250, activation='relu'))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
#
# #model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=2, batch_size=128, verbose=1)
# #results = model.evaluate(x_test, y_test, verbose=0)
# estimator = KerasClassifier(build_fn=baselineModel, epochs=2, batch_size=128, verbose=0)
# kfold = StratifiedKFold(n_splits=n_splits, random_state=0)
# results = cross_val_score(estimator, x, encoded_y, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))


# convolutional neural network 2D using MNIST dataset

# seed = 0
# np.random.seed(seed)
#
# # load data
# (X_train, y_train), (X_test, y_test) = mnist.load_data()
# # reshape to be [samples][pixels][width][height]
# X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
# X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
#
# # normalize inputs from 0-255 to 0-1
# X_train = X_train / 255
# X_test = X_test / 255
#
# # one hot encode outputs
# y_train = np_utils.to_categorical(y_train)
# y_test = np_utils.to_categorical(y_test)
# num_classes = y_test.shape[1]
#
#
# def baselineModel():
#     model = Sequential()
#     model.add(Convolution2D(32, 5, 5, border_mode='valid', input_shape=(1, 28, 28), activation='relu'))
#     model.add(MaxPooling2D(pool_size=(2,2)))
#     model.add(Dropout(0.2))
#     model.add(Flatten())
#     model.add(Dense(128, activation='relu'))
#     model.add(Dense(num_classes, activation='softmax'))
#     model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model
#
# model = baselineModel()
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=200, verbose=2)
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))





################### 1D Conv Network working well ############################################
# from __future__ import print_function
#
# from keras.preprocessing import sequence
# from keras.models import Sequential
# from keras.layers import Dense, Dropout, Activation
# from keras.layers import Embedding
# from keras.layers import Conv1D, GlobalMaxPooling1D
# from keras.datasets import imdb
#
# # set parameters:
# max_features = 5000
# maxlen = 400
# batch_size = 32
# embedding_dims = 50
# filters = 250
# kernel_size = 3
# hidden_dims = 250
# epochs = 2
#
# print('Loading data...')
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
# print(len(x_train), 'train sequences')
# print(len(x_test), 'test sequences')
#
# print('Pad sequences (samples x time)')
# x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
# x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
# print('x_train shape:', x_train.shape)
# print('x_test shape:', x_test.shape)
#
# print('Build model...')
# model = Sequential()
#
# # we start off with an efficient embedding layer which maps
# # our vocab indices into embedding_dims dimensions
# model.add(Embedding(max_features,
#                     embedding_dims,
#                     input_length=maxlen))
# model.add(Dropout(0.2))
#
# # we add a Convolution1D, which will learn filters
# # word group filters of size filter_length:
# model.add(Conv1D(filters,
#                  kernel_size,
#                  padding='valid',
#                  activation='relu',
#                  strides=1))
# # we use max pooling:
# model.add(GlobalMaxPooling1D())
#
# # We add a vanilla hidden layer:
# model.add(Dense(hidden_dims))
# model.add(Dropout(0.2))
# model.add(Activation('relu'))
#
# # We project onto a single unit output layer, and squash it with a sigmoid:
# model.add(Dense(1))
# model.add(Activation('sigmoid'))
#
# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])
# model.fit(x_train, y_train,
#           batch_size=batch_size,
#           epochs=epochs,
# validation_data=(x_test, y_test))
