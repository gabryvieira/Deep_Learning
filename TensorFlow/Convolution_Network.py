import numpy as np
import time
from keras.layers import Dense, Conv1D, MaxPooling1D, Dropout, Flatten
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import matthews_corrcoef, make_scorer
from sklearn.preprocessing import LabelEncoder

seed = 7
np.random.seed(seed)

# global variables

dna_sequence = '../Datasets/exovar_sequence_window.txt'

file_seq_orig = "dna_sequences/orig_seq_humvar.txt"
outfile_seq_orig = "dna_sequences/origOut_seq_humvar.txt"

file_seq_mut = "dna_sequences/mut_seq_humvar.txt"
outfile_seq_mut = "dna_sequences/mutOut_seq_humvar.txt"

n_splits = 10
epochs = 15
batch_size = 32
kernel_size = 3
num_strides = 1
filters = 32

m = {
    'A': np.array([1, 0, 0, 0, 0]),
    'C': np.array([0, 1, 0, 0, 0]),
    'G': np.array([0, 0, 1, 0, 0]),
    'T': np.array([0, 0, 0, 1, 0]),
    'X': np.array([0, 0, 0, 0, 1])
}

num_of_nucleotides = len(m['A'])
n_chars = 201

mcc_scorer = make_scorer(matthews_corrcoef)
scoring = mcc_scorer



def readSequence(file):
    file_data = np.genfromtxt(file, delimiter=',', skip_header=0, dtype='str') # str type for sequence and geneID
    #original_seq = np.ravel(file_data[:, 2])
    #mutated_sequence = np.ravel(file_data[:, 3])
    original_seq = file_data[:,2]
    mutated_sequence = file_data[:,3]

    #writeSequences(original_seq, mutated_sequence)

    file_data_gene = np.ravel(file_data[:, 0])
    geneID = [i.split(':', 1)[0] for i in file_data_gene]  # remove : and everything after in array
    file_class = np.genfromtxt(file, delimiter=',', skip_header=0) # float type
    true_class = np.ravel(file_class[:, 1])  # get all lines
    positive_fraction = (sum(true_class) * 100.0) / len(true_class)
    print("Done! %i data points were read, with %i features (%.2f%% positive)" % (len(true_class), len(file_data[0, :-1]), positive_fraction))
    return geneID, true_class, original_seq, mutated_sequence



def writeSequences(original, mutated):
    orig_seq_file = open(file_seq_orig, 'w')
    for item in original:
        orig_seq_file.write("%s\n" % item)
    orig_seq_file.close()

    mut_seq_file = open(file_seq_mut, 'w')
    for item in mutated:
        mut_seq_file.write("%s\n" % item)
    mut_seq_file.close()



def pre_processing(orig_seq, mut_seq):



    a = np.array([np.array([m[c] for c in seq]) for seq in orig_seq])
    b = np.array([np.array([m[c] for c in seq]) for seq in mut_seq])

    #a = a.reshape((orig_seq.shape[0], num_of_nucleotides, 5))
    seq = np.subtract(a,b)
    #seq[seq < 0] = 0
    #seq = np.vstack((a,b))
    #seq = seq.reshape((orig_seq.shape[0], n_chars*2, num_of_nucleotides))
    #print("After reshape")
    print(seq)
    return seq



def baseLineModel():
    model = Sequential()
    model.add(Conv1D(filters=filters, kernel_size=kernel_size, input_shape=(n_chars, num_of_nucleotides), padding='same', activation='relu', strides=num_strides))
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
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy']) # train the model
    return model



def main():
    start_time = time.time()
    gene, class_y, orig_seq, mut_seq = readSequence(dna_sequence)
    seq_bin = pre_processing(orig_seq, mut_seq)


    encoder = LabelEncoder()
    encoder.fit(class_y)
    class_y = encoder.transform(class_y)

    execTime = (time.time() - start_time)
    print("Pre-processing time: %s seconds" % execTime)


    estimators = [('cnn', KerasClassifier(build_fn=baseLineModel, epochs=epochs, batch_size=batch_size, verbose=0))]
    #estimators = KerasClassifier(build_fn=baseLineModel, epochs=epochs, batch_size=batch_size, verbose=0)
    pipeline = Pipeline(estimators)
    #kfold = GroupKFold(n_splits=n_splits)
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=6)
    training_time = time.time()
    results = cross_val_score(pipeline, seq_bin, class_y, groups=gene, cv=kfold, scoring=scoring)
    exec_time = (time.time() - training_time)
    print("Classification time:  %s seconds" % exec_time)
    print("Accuracy: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    classtime = time.time()
    predicted = cross_val_predict(pipeline, seq_bin, class_y, groups=gene, cv=kfold)
    execTime = (time.time() - classtime)
    print("Classification time:  %s seconds" % execTime)
    print(predicted)

if __name__ == '__main__':
    main()