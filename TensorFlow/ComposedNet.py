import numpy as np
from keras.engine import Input
from keras.engine import Model
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten, Merge, merge, concatenate
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.model_selection import cross_val_score
import time
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import LabelEncoder


seed = 7
np.random.seed(seed)

# global variables
dna_sequence = '../Datasets/varibench_codon_sequence.txt'
dna_sequence_code = '../Datasets/exovar_equence.txt'



np_splits = 10
epochs = 20
batch_size = 32
kernel_size = 3
num_strides = 1
filters = 32

# 183 for predictSNP, swissvar and varibench  201 for exovar and humvar usando o dicionario m
# aquando do uso dos dados por codoes, por defeito o valor deve ser igualado a 61
num_codons = 183

scoring = 'roc_auc'




######## Para correr a estrategia Physical-chemical properties, descomentar tudo ate a linha 173

# properties = ['hydropathy', 'polarity', 'averageMass', 'isoElectricPt', 'vanDerWaals', 'netCharge', 'pK1', 'pK2', 'pKa']
#
# aa_properties = {'hydropathy': {'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8, 'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8, 'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5, 'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3},
#                  'polarity': {'A': 8.1, 'C': 5.5, 'D': 13, 'E': 10.5, 'F': 5.2, 'G': 9, 'H': 10.4, 'I': 5.2, 'K': 11.3, 'L': 4.9, 'M': 5.7, 'N': 11.6, 'P': 8, 'Q': 12.3, 'R': 10.5, 'S': 9.2, 'T': 8.6, 'V': 5.9, 'W': 5.4, 'Y': 6.2},
#                  'averageMass': {'A': 89.1, 'C': 121.2, 'D': 133.1, 'E': 147.1, 'F': 165.2, 'G': 75.1, 'H': 155.2, 'I': 131.2, 'K': 146.2, 'L': 131.2, 'M': 149.2, 'N': 132.1, 'P': 115.1, 'Q': 146.1, 'R': 174.2, 'S': 105.1, 'T': 119.1, 'V': 117.1, 'W': 204.2, 'Y': 181.2},
#                  'isoElectricPt': {'A': 6.01, 'C': 5.05, 'D': 2.85, 'E': 3.15, 'F': 5.49, 'G': 6.06, 'H': 7.6, 'I': 6.05, 'K': 9.6, 'L': 6.01, 'M': 5.74, 'N': 5.41, 'P': 6.3, 'Q': 5.65, 'R': 10.76, 'S': 5.68, 'T': 5.6, 'V': 6, 'W': 5.89, 'Y': 5.64},
#                  'vanDerWaals': {'A': 67, 'C': 86, 'D': 91, 'E': 109, 'F': 135, 'G': 48, 'H': 118, 'I': 124, 'K': 135, 'L': 124, 'M': 124, 'N': 96, 'P': 90, 'Q': 114, 'R': 148, 'S': 73, 'T': 93, 'V': 105, 'W': 163, 'Y': 141},
#                  'netCharge': {'A': 0.007, 'C': -0.037, 'D': -0.024, 'E': 0.007, 'F': 0.038, 'G': 0.179, 'H': -0.011, 'I': 0.022, 'K': 0.018, 'L': 0.052, 'M': 0.003, 'N': 0.005, 'P': 0.24, 'Q': 0.049, 'R': 0.044, 'S': 0.005, 'T': 0.003, 'V': 0.057, 'W': 0.038, 'Y': 0.024},
#                  'pK1': {'A': 2.35, 'C': 1.92, 'D': 1.99, 'E': 2.1, 'F': 2.2, 'G': 2.35, 'H': 1.8, 'I': 2.32, 'K': 2.16, 'L': 2.33, 'M': 2.13, 'N': 2.14, 'P': 1.95, 'Q': 2.17, 'R': 1.82, 'S': 2.19, 'T': 2.09, 'V': 2.39, 'W': 2.46, 'Y': 2.2},
#                  'pK2': {'A': 9.87, 'C': 10.7, 'D': 9.9, 'E': 9.47, 'F': 9.31, 'G': 9.78, 'H': 9.33, 'I': 9.76, 'K': 9.06, 'L': 9.74, 'M': 9.28, 'N': 8.72, 'P': 10.64, 'Q': 9.13, 'R': 8.99, 'S': 9.21, 'T': 0.0, 'V': 9.74, 'W': 9.41, 'Y': 9.21},
#                  'pKa': {'A': 0.0, 'C': 8.18, 'D': 3.9, 'E': 4.07, 'F': 0.0, 'G': 0.0, 'H': 6.04, 'I': 0.0, 'K': 10.54, 'L': 0.0, 'M': 0.0, 'N': 5.41, 'P': 0.0, 'Q': 0.0, 'R': 12.48, 'S': 5.68, 'T': 5.53, 'V': 0.0, 'W': 5.885, 'Y': 10.46}
#                  }


# start_time = time.time()
#
# codon_aa_map = dict()
# for line in open('gct.txt', 'rt'):
#     aa, _, _, codons = line.strip('\n').split('\t')
#     for codon in codons.split('|'):
#             codon_aa_map[codon] = aa
#
#
# codon_aa_map['XXX'] = '0'
# print(codon_aa_map)
# for p in properties:
#     aa_properties[p]['0']=100   # pode experimentar-se com outro valor, p.ex ‘-100'
#
# for p in properties:
#     aa_properties[p]['*']=100   # pode experimentar-se com outro valor, p.ex ‘-100'
#
#
# filename = '../Datasets/varibench_codon_sequence.txt'
#
#
# y = np.ravel(np.genfromtxt(filename, delimiter=',', skip_header=0, usecols=1))
# y[y==-1]=0 # os -1 colocados a 0
#
# file_data = np.genfromtxt(filename, delimiter=',', skip_header=0, dtype='S')
# groups = file_data[:, 0] # geneID
#
#
#
# N = len(y)
# print(N)
# M = len(properties)  # numero de propriedades (= tamanho do vetor que representa cada AA)
# print(M)
# pos = 30  # posicao do codao com mutacao, nos ficheiros de dados
# W = 3  # janela de codoes / AAs a usar
#
#
# #sequencia original
# X1 = np.array([np.array([[aa_properties[p][codon_aa_map[codon]] for p in properties]
#                          for codon in x.decode().split('.')[pos-W:pos+W+1]])
#                             for x in file_data[:, 2]])#.reshape(N, M*(2*W+1))
#
# # sequencia mutada
# X2 = np.array([np.array([[aa_properties[p][codon_aa_map[codon]] for p in properties]
#                          for codon in x.decode().split('.')[pos-W:pos+W+1]])
#                             for x in file_data[:, 3]])#.reshape(N, M*(2*W+1))
#
# # diferenca
# #X = X1 - X2
# execTime = (time.time() - start_time)
# print("Pre-processing time: %s seconds" % execTime)
#
# model1 = Sequential()
# model1.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(7, M), padding='same',
#                   activation='relu'))
# model1.add(MaxPooling1D(pool_size=2))
# model1.add(Dropout(0.2))
# model1.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
# model1.add(MaxPooling1D(pool_size=2))
# model1.add(Dropout(0.2))
# model1.add(Flatten())
# model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# model1.add(Dropout(0.2))
# model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# model1.add(Dropout(0.2))
# model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))
#
# #model1.add(Dropout(0.2))
# #model1.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# #model1.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# #model1.fit(orig_seq, class_y, epochs=epochs, batch_size=batch_size, verbose=1)
# #print(model1.predict(mut_seq))
#
# input1 = Input(shape=(7, M))
# seq1 = model1(input1)
#
# model2 = Sequential()
# model2.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(7, M), padding='same',
#                   activation='relu'))
# model2.add(MaxPooling1D(pool_size=2))
# model2.add(Dropout(0.2))
# model2.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
# model2.add(MaxPooling1D(pool_size=2))
# model2.add(Dropout(0.2))
# model2.add(Flatten())
# model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# model2.add(Dropout(0.2))
# model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# model2.add(Dropout(0.2))
# model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))
# #model2.add(Dropout(0.2))
# #model2.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
# #model2.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])  #rmsprop
# #model2.fit(mut_seq, class_y, epochs=epochs, batch_size=batch_size, verbose=1)
# #print(model2.predict(mut_seq))
# input2 = Input(shape=(7, M))
# seq2 = model2(input2)
#
# merged = concatenate([seq1, seq2])
# # merged = merge([model1, model2], mode='concat')
# x = Dense(100, activation='relu')(merged)
# x = Dense(64, activation='relu')(x)
# x = Dense(1, activation='sigmoid')(x)
# model = Model(inputs=[input1, input2], output=x)
# model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# training_time = time.time()
# model.fit([X1, X2], y, epochs=epochs, batch_size=batch_size, verbose=1)
# exec_time = (time.time() - training_time)
# print("Training time:  %s seconds" % exec_time)
# classtime = time.time()
# model.predict([X1, X2])
# execTime = (time.time() - classtime)
# print("Classification time:  %s seconds" % execTime)
# print(model.evaluate([X1, X2], y, batch_size=batch_size))














## Restantes estrategias ######

#### One-Hot vector na sequencia de DNA sem ser no formato de codoes #####

m = {
    'A': np.array([1, 0, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0, 0]),
    'G': np.array([0, 0, 1, 0, 0]),
    'T': np.array([0, 0, 0, 1, 0]),
    'X': np.array([0, 0, 0, 0, 1])
}

numOf_digits = len(m['A'])




########### 65 One-Hot vector $#################################
# codons = ["TTT", "TTC", "TTA", "TTG", "TCT", "TCC", "TCA", "TCG", "TAT", "TAC", "TAA", "TAG", "TGT", "TGC", "TGA",
#     "TGG", "CTT", "CTC", "CTA", "CTG", "CCT", "CCC", "CCA", "CCG", "CAT", "CAC", "CAA", "CAG", "CGT", "CGC", "CGA",
#     "CGG", "ATT", "ATC", "ATA", "ATG", "ACT", "ACC", "ACA", "ACG", "AAT", "AAC", "AAA", "AAG", "AGT", "AGC", "AGA",
#     "AGG", "GTT", "GTC", "GTA", "GTG", "GCT", "GCC", "GCA", "GCG", "GAT", "GAC", "GAA", "GAG", "GGT", "GGC", "GGA",
#     "GGG", "XXX"]
#
# numOf_digits = len(codons)
#
# #criacao dos 64 numpy arrays a zero e sua colocacao do 1 no sitio correcto
# cod = []
# for i in range(0, 65):
#     cod.append(np.zeros(65))
#     np.put(cod[i], i, 1)
#
# dictionary = dict(zip(codons, cod))

###########################################################


########## amino acid sequence ############################
# map = {
#     "TTT": np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "TTC": np.array([0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "TTA": np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]), "TTG": np.array([0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0]),
#     "TCT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]), "TCC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]), "TCA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]), "TCG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
#     "TAT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]), "TAC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
#     "TAA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]), "TAG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
#     "TGT": np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "TGC": np.array([0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "TGA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]),
#     "TGG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0]),
#     "CTT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]), "CTC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]), "CTA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]), "CTG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0]),
#     "CCT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]), "CCC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]), "CCA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]), "CCG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0]),
#     "CAT": np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "CAC": np.array([0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "CAA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]), "CAG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]),
#     "CGT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), "CGC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), "CGA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), "CGG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
#     "ATT": np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]), "ATC": np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]), "ATA": np.array([0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "ATG": np.array([0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0]),
#     "ACT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), "ACC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), "ACA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]), "ACG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0]),
#     "AAT": np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "AAC": np.array([0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "AAA": np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]), "AAG": np.array([0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "AGT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]), "AGC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]),
#     "AGA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]), "AGG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0]),
#     "GTT": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]), "GTC": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]), "GTA": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]), "GTG": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]),
#     "GCT": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GCC": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GCA": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GCG": np.array([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "GAT": np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GAC": np.array([0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "GAA": np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GAG": np.array([0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "GGT": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GGC": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GGA": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), "GGG": np.array([0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]),
#     "XXX": np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0])}
#
# numOf_digits = len(map['CTT'])
##################################################################################

##########  amino acid property (1 and 0) ###################################################
# amino_acid_pro = {
#     "TTT": np.array([0,0,0,0,0,1,0,0,1,0]), "TTC": np.array([0,0,0,0,0,1,0,0,1,0]),
#     "TTA": np.array([0,0,0,0,1,1,0,1,0,0]), "TTG": np.array([0,0,0,0,1,1,0,1,0,0]),
#     "TCT": np.array([0,0,0,1,0,0,1,0,0,0]), "TCC": np.array([0,0,0,1,0,0,1,0,0,0]), "TCA": np.array([0,0,0,1,0,0,1,0,0,0]), "TCG": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "TAT": np.array([0,0,0,0,0,1,0,0,1,0]), "TAC": np.array([0,0,0,0,0,1,0,0,1,0]),
#     "TGT": np.array([0,0,0,0,0,0,1,0,0,0]), "TGC": np.array([0,0,0,0,0,0,1,0,0,0]),
#     "TGA": np.array([0,0,0,0,0,0,0,0,0,0]), "TAA": np.array([0,0,0,0,0,0,0,0,0,0]), "TAG": np.array([0,0,0,0,0,0,0,0,0,0]), # stop codons
#     "TGG": np.array([0,0,0,0,0,1,0,0,1,0]),
#     "CTT": np.array([0,0,0,0,1,1,0,1,0,0]), "CTC": np.array([0,0,0,0,1,1,0,1,0,0]), "CTA": np.array([0,0,0,0,1,1,0,1,0,0]), "CTG": np.array([0,0,0,0,1,1,0,1,0,0]),
#     "CCT": np.array([0,0,0,1,0,0,1,0,0,0]), "CCC": np.array([0,0,0,1,0,0,1,0,0,0]), "CCA": np.array([0,0,0,1,0,0,1,0,0,0]), "CCG": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "CAT": np.array([0,0,0,0,0,1,0,0,1,0]), "CAC": np.array([0,0,0,0,0,1,0,0,1,0]),
#     "CAA": np.array([1,0,0,0,0,1,0,0,0,0]), "CAG": np.array([1,0,0,0,0,1,0,0,0,0]),
#     "CGT": np.array([0,0,1,0,0,0,0,0,0,0]), "CGC": np.array([0,0,1,0,0,0,0,0,0,0]), "CGA": np.array([0,0,1,0,0,0,0,0,0,0]), "CGG": np.array([0,0,1,0,0,0,0,0,0,0]),
#     "ATT": np.array([0,0,0,0,1,1,0,1,0,0]), "ATC": np.array([0,0,0,0,1,1,0,1,0,0]), "ATA": np.array([0,0,0,0,1,1,0,1,0,0]),
#     "ATG": np.array([0,0,0,0,1,1,0,0,0,0]),
#     "ACT": np.array([0,0,0,1,0,0,1,0,0,0]), "ACC": np.array([0,0,0,1,0,0,1,0,0,0]), "ACA": np.array([0,0,0,1,0,0,1,0,0,0]), "ACG": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "AAT": np.array([1,0,0,0,0,0,1,0,0,0]), "AAC": np.array([1,0,0,0,0,0,1,0,0,0]),
#     "AAA": np.array([0,0,1,0,0,1,0,0,0,0]), "AAG": np.array([0,0,1,0,0,1,0,0,0,0]),
#     "AGT": np.array([0,0,0,1,0,0,1,0,0,0]), "AGC": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "AGA": np.array([0,0,1,0,0,0,0,0,0,0]), "AGG": np.array([0,0,1,0,0,0,0,0,0,0]),
#     "GTT": np.array([0,0,0,0,1,0,1,1,0,0]), "GTC": np.array([0,0,0,0,1,0,1,1,0,0]), "GTA": np.array([0,0,0,0,1,0,1,1,0,0]), "GTG": np.array([0,0,0,0,1,0,1,1,0,0]),
#     "GCT": np.array([0,0,0,1,0,0,1,0,0,0]), "GCC": np.array([0,0,0,1,0,0,1,0,0,0]), "GCA": np.array([0,0,0,1,0,0,1,0,0,0]), "GCG": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "GAT": np.array([1,1,0,0,0,0,1,0,0,0]), "GAC": np.array([1,1,0,0,0,0,1,0,0,0]),
#     "GAA": np.array([1,1,0,0,0,1,0,0,0,0]), "GAG": np.array([1,1,0,0,0,1,0,0,0,0]),
#     "GGT": np.array([0,0,0,1,0,0,1,0,0,0]), "GGC": np.array([0,0,0,1,0,0,1,0,0,0]), "GGA": np.array([0,0,0,1,0,0,1,0,0,0]), "GGG": np.array([0,0,0,1,0,0,1,0,0,0]),
#     "XXX": np.array([0,0,0,0,0,0,0,0,0,1])
# }
#
# numOf_digits = len(amino_acid_pro['CTT'])
##################################################################################################################




###############################################################################################################
# vennStrat = {
#     "TTT": np.array([0,1,0,0,1,0,0,0,0,0,0]), "TTC": np.array([0,1,0,0,1,0,0,0,0,0,0]),
#     "TTA": np.array([1,0,0,0,1,0,0,0,0,0,0]), "TTG": np.array([1,0,0,0,1,0,0,0,0,0,0]),
#     "TCT": np.array([0,0,0,0,0,1,1,0,1,0,1]), "TCC": np.array([0,0,0,0,0,1,1,0,1,0,1]), "TCA": np.array([0,0,0,0,0,1,1,0,1,0,1]), "TCG": np.array([0,0,0,0,0,1,1,0,1,0,1]),
#     "TAT": np.array([0,1,0,0,0,0,1,0,0,0,0]), "TAC": np.array([0,1,0,0,0,0,1,0,0,0,0]),
#     "TAA": np.array([0,0,0,0,0,0,0,0,0,0,0]), "TAG": np.array([0,0,0,0,0,0,0,0,0,0,0]), "TGA": np.array([0,0,0,0,0,0,0,0,0,0,0]),
#     "TGT": np.array([0,0,0,0,1,0,1,0,1,1,1]), "TGC": np.array([0,0,0,0,1,0,1,0,1,1,1]),
#     "TGG": np.array([0,1,0,0,0,0,1,0,0,0,0]),
#     "CTT": np.array([1,0,0,0,1,0,0,0,0,0,0]), "CTC": np.array([1,0,0,0,1,0,0,0,0,0,0]), "CTA": np.array([1,0,0,0,1,0,0,0,0,0,0]), "CTG": np.array([1,0,0,0,1,0,0,0,0,0,0]),
#     "CCT": np.array([0,0,0,0,0,0,0,0,1,0,0]), "CCC": np.array([0,0,0,0,0,0,0,0,1,0,0]), "CCA": np.array([0,0,0,0,0,0,0,0,1,0,0]), "CCG": np.array([0,0,0,0,0,0,0,0,1,0,0]),
#     "CAT": np.array([0,1,0,1,1,0,1,1,0,0,0]), "CAC": np.array([0,1,0,1,1,0,1,1,0,0,0]),
#     "CAA": np.array([0,0,1,0,0,0,1,0,0,0,0]), "CAG": np.array([0,0,1,0,0,0,1,0,0,0,0]),
#     "CGT": np.array([0,0,0,1,0,0,1,1,0,0,0]), "CGC": np.array([0,0,0,1,0,0,1,1,0,0,0]), "CGA": np.array([0,0,0,1,0,0,1,1,0,0,0]), "CGG": np.array([0,0,0,1,0,0,1,1,0,0,0]),
#     "ATT": np.array([1,0,0,0,1,0,0,0,0,0,0]), "ATC": np.array([1,0,0,0,1,0,0,0,0,0,0]), "ATA": np.array([1,0,0,0,1,0,0,0,0,0,0]),
#     "ATG": np.array([0,0,0,0,1,0,0,0,0,1,0]),
#     "ACT": np.array([0,0,0,0,0,1,1,0,1,0,0]), "ACC": np.array([0,0,0,0,0,1,1,0,1,0,0]), "ACA": np.array([0,0,0,0,0,1,1,0,1,0,0]), "ACG": np.array([0,0,0,0,0,1,1,0,1,0,0]),
#     "AAT": np.array([0,0,1,0,0,0,1,0,1,0,0]), "AAC": np.array([0,0,1,0,0,0,1,0,1,0,0]),
#     "AAA": np.array([0,0,0,1,1,0,1,1,0,0,0]), "AAG": np.array([0,0,0,1,1,0,1,1,0,0,0]),
#     "AGT": np.array([0,0,0,0,0,1,1,0,1,0,1]), "AGC": np.array([0,0,0,0,0,1,1,0,1,0,1]),
#     "AGA": np.array([0,0,0,1,0,0,1,1,0,0,0]), "AGG": np.array([0,0,0,1,0,0,1,1,0,0,0]),
#     "GTT": np.array([1,0,0,0,1,0,0,0,1,0,0]), "GTC": np.array([1,0,0,0,1,0,0,0,1,0,0]), "GTA": np.array([1,0,0,0,1,0,0,0,1,0,0]), "GTG": np.array([1,0,0,0,1,0,0,0,1,0,0]),
#     "GCT": np.array([0,0,0,0,1,0,0,0,1,0,1]), "GCC": np.array([0,0,0,0,1,0,0,0,1,0,1]), "GCA": np.array([0,0,0,0,1,0,0,0,1,0,1]), "GCG": np.array([0,0,0,0,1,0,0,0,1,0,1]),
#     "GAT": np.array([0,0,0,1,0,0,1,0,1,0,0]), "GAC": np.array([0,0,0,1,0,0,1,0,1,0,0]),
#     "GAA": np.array([0,0,0,1,0,0,1,0,0,0,0]), "GAG": np.array([0,0,0,1,0,0,1,0,0,0,0]),
#     "GGT": np.array([0,0,0,0,0,0,0,0,1,0,1]), "GGC": np.array([0,0,0,0,0,0,0,0,1,0,1]), "GGA": np.array([0,0,0,0,0,0,0,0,1,0,1]), "GGG": np.array([0,0,0,0,0,0,0,0,1,0,1]),
#     "XXX": np.array([1,1,1,1,1,1,1,1,1,1,1])}
#
# numOf_digits = len(vennStrat['CTT'])

def readSequence(file):

    ############# para a sequencia DNA "seguida" ##########################
    seq = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str')
    original_seq = np.ravel(seq[:, 2])
    mut_seq = np.ravel(seq[:, 3])
    original_codon_seq = np.array([str(i).replace(".", "") for i in original_seq])
    mutated_codon_seq = np.array([str(i).replace(".", "") for i in mut_seq])

    # para reduzir janela de codoes #
    # original_codon_seq = original_codon_seq[:, (30-int(num_codons/2)):(31+int(num_codons/2)):1] 
    # mutated_codon_seq = mutated_codon_seq[:, (30-int(num_codons/2)):(31+int(num_codons/2)):1]


    ########### para a sequencia DNA por codoes """"""""""""""""""""""""""""""
    # sequencia original
    # original_codon_seq = np.genfromtxt(file, delimiter='.', skip_header=0, dtype='str', usecols=range(2, 63))
    # original_codon_seq[:, 0] = [i.split('1,', 1)[1] for i in
    #                             original_codon_seq[:, 0]]  # eliminar o que esta antes do 1o codon
    # original_codon_seq[:, 60] = [i.split(',', 1)[0] for i in
    #                              original_codon_seq[:, 60]]  # eliminar codon aseguir a virgula no ultimo codao
    #
    # original_codon_seq = original_codon_seq[:, (30-int(num_codons/2)):(31+int(num_codons/2)):1] 
    # mutated_codon_seq = mutated_codon_seq[:, (30-int(num_codons/2)):(31+int(num_codons/2)):1]
    #
    # # sequencia mutada
    # mutated_codon_seq = np.genfromtxt(file, delimiter='.', skip_header=0, dtype='str', usecols=range(62, 123))
    # mutated_codon_seq[:, 0] = [i.split(',', 1)[1] for i in mutated_codon_seq[:, 0]]  # obter primeiro codao da sequencia

    ###############################################################################################

    file_data = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str')  # str type for sequence and geneID
    file_data_gene = np.ravel(file_data[:, 0])
    geneID = [i.split(':', 1)[0] for i in file_data_gene]  # remove : and everything after in array
    file_class = np.genfromtxt(file, delimiter=',', skip_header=0)  # float type
    true_class = np.ravel(file_class[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)
    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (len(true_class), len(file_data[0, :-1]), positive_fraction))
    return geneID, true_class, original_codon_seq, mutated_codon_seq



def pre_processing(orig_seq, mut_seq):

    a = np.array([np.array([m[c] for c in seq]) for seq in orig_seq]) # seq original
    b = np.array([np.array([m[c] for c in seq]) for seq in mut_seq])  # seq mutada

    #seq = np.subtract(a,b)
    #seq[seq < 0] = 0
    return a,b



def main():
    start_time = time.time()
    gene, class_y, orig_seq, mut_seq = readSequence(dna_sequence)
    orig_seq, mut_seq = pre_processing(orig_seq, mut_seq)

    encoder = LabelEncoder()
    encoder.fit(class_y)
    class_y = encoder.transform(class_y)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)

    # train test split aqui !!!
    X_train, X_test, y_train, y_test = train_test_split(orig_seq, class_y, test_size=0.5)
    X_train2, X_test2, y_train2, y_test2 = train_test_split(mut_seq, class_y, test_size=0.5)

    # CNN1
    model1 = Sequential()
    model1.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(num_codons, numOf_digits), padding='same',
                      activation='relu'))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Dropout(0.2))
    model1.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model1.add(MaxPooling1D(pool_size=2))
    model1.add(Dropout(0.2))
    model1.add(Flatten())
    model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model1.add(Dropout(0.2))
    model1.add(Dense(100, kernel_initializer='uniform', activation='relu'))

    input1 = Input(shape=(num_codons, numOf_digits))
    seq1 = model1(input1)


    #CNN2
    model2 = Sequential()
    model2.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(num_codons, numOf_digits), padding='same',
                      activation='relu'))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Dropout(0.2))
    model2.add(Conv1D(filters=filters, kernel_size=kernel_size, padding='same', activation='relu'))
    model2.add(MaxPooling1D(pool_size=2))
    model2.add(Dropout(0.2))
    model2.add(Flatten())
    model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))
    model2.add(Dropout(0.2))
    model2.add(Dense(100, kernel_initializer='uniform', activation='relu'))

    input2 = Input(shape=(num_codons, numOf_digits))
    seq2 = model2(input2)

    merged = concatenate([seq1, seq2]) # juntar as duas para ligar a NN
    
    # Neural Network
    x = Dense(100, activation='relu')(merged)
    x = Dense(64, activation='relu')(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input1, input2], output=x)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	training_time = time.time()
    model.fit([X_train, X_train2], y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    exec_time = (time.time() - training_time)
    print("Training time:  %s seconds" % exec_time)
    classtime = time.time()
    pred_proba = model.predict([X_test, X_test2], batch_size=batch_size)
    #print(pred_proba)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(roc_auc_score(y_test, pred_proba))
    

if __name__ == '__main__':
    main()