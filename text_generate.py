import nltk
import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

file = open("wonderland.txt").read()
file

input = file.lower()
tokenizer = RegexpTokenizer(r'\w+')
tokens = tokenizer.tokenize(input)
filtered = filter(lambda token: token not in stopwords.words('english'), tokens)
inputs_process=" ".join(filtered)

chars = sorted(list(set(inputs_process)))
char_to_number = dict((c, i) for i, c in enumerate(chars))

noc = len(inputs_process)
print ("Total number of characters:", noc)
vocab= len(chars)
print ("Total vocab:", vocab)

sequence_l = 100
x = []
y = []

for i in range(0, noc - sequence_l, 1):
    in_seq = inputs_process[i:i + sequence_l]
    out_seq = inputs_process[i + sequence_l]
    
    x.append([char_to_number[char] for char in in_seq])
    y.append(char_to_number[out_seq])

total_patterns = len(x)
print ("Total Patterns:", total_patterns)

X = numpy.reshape(x, (total_patterns, sequence_l, 1))
X = X/float(vocab)
Y = np_utils.to_categorical(y)

model = Sequential()
model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dropout(0.2))
model.add(Dense(Y.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam')

path = "weights_model.hdf5"
checkpoint = ModelCheckpoint(path, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

model.fit(X, Y, epochs=20, batch_size=128, callbacks=callbacks)
file = "weights_model.hdf5"
model.load_weights(file)
model.compile(loss='categorical_crossentropy', optimizer='adam')

number_to_char = dict((i, c) for i, c in enumerate(chars))

start = numpy.random.randint(0, len(x) - 1)
pattern = x[start]
print("The Random Seed:")
print("\"", ''.join([number_to_char[value] for value in pattern]), "\"")

for i in range(1000):
    x = numpy.reshape(pattern, (1, len(pattern), 1))
    x = x / float(vocab)
    prediction = model.predict(x, verbose=0)
    index = numpy.argmax(prediction)
    result = number_to_char[index]
    seq_in = [number_to_char[value] for value in pattern]

    sys.stdout.write(result)

    pattern.append(index)
    pattern = pattern[1:len(pattern)]