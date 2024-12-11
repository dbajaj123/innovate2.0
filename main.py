from keras import preprocessing
import pandas as pd
import numpy as np


data = pd.read_csv('data.csv')

x_train = data['message']

df_encoded = pd.get_dummies(data["species"])

new_df = np.array(data["tail"].apply(lambda x: (x=="yes")*1))

dl = np.array(data["fingers"])



y_train = np.array(df_encoded)*1
print(y_train, y_train.shape)

def unisonShuffleDataset(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
max_len=100
max_words=10000

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)
word_index = tokenizer.word_index
print(len(word_index))

data = pad_sequences(sequences, maxlen=max_len)
x_train = data.T

x_train = np.vstack((x_train, dl))
x_train = np.vstack((x_train, new_df)).T

print(x_train, x_train.shape)



from keras.layers import LSTM, Embedding, Dense, SimpleRNN
from keras.models import Sequential


model=Sequential()
model.add(Embedding(10000, 1024))
model.add(LSTM(1024))
model.add(Dense(10, activation = 'sigmoid'))
model.compile(optimizer="rmsprop", loss="binary_crossentropy", metrics=["acc"])
print(x_train, x_train.shape)
print(y_train)
history=model.fit(x_train, y_train, epochs=10, batch_size=8, validation_split=0.1)
