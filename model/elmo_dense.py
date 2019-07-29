import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, LSTM, Bidirectional, BatchNormalization,Embedding, Input, Dropout, TimeDistributed, Concatenate,Flatten, Lambda, GlobalMaxPool1D
from keras.models import Model, load_model
from keras.engine import Layer
import keras
import numpy as np
import pickle
from collections import Counter
import json

def load_data(file_name, sample_ratio=1, n_class=15, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    x = csv_file["normalize_heading"].values
    y = csv_file["price_label"].values

    return x, y


# In[3]:


x_train, y_train = load_data("../data/train.csv", sample_ratio=1e-2, one_hot=False)
x_test, y_test = load_data("./data/test.csv", one_hot=False)


# In[4]:


# parameter of max word length
time_steps = 100


# building vocabulary from dataset
def build_vocabulary(sentence_list):
    unique_words = " ".join(sentence_list).strip().split()
    word_count = Counter(unique_words).most_common()
    vocabulary = {}
    for word, _ in word_count:
        vocabulary[word] = len(vocabulary)        

    return vocabulary



# In[11]:



# Create datasets (Only take up to time_steps words for memory)
train_text = x_train.tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text, dtype=object)[:, np.newaxis]
train_label = y_train

test_text = x_test.tolist()
test_text = [' '.join(t.split()[0:time_steps]) for t in test_text]
test_text = np.array(test_text, dtype=object)[:, np.newaxis]
test_label = y_test

# Create a custom layer that allows us to update weights (lambda layers do not have trainable parameters!)

class ElmoEmbeddingLayer(Layer):
    def __init__(self, **kwargs):
        self.dimensions = 1024
        self.trainable=True
        super(ElmoEmbeddingLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
                               name="{}_module".format(self.name))

        self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
        super(ElmoEmbeddingLayer, self).build(input_shape)

    def call(self, x, mask=None):
        result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
                      as_dict=True,
                      signature='default',
                      )['default']
        return result

    def compute_mask(self, inputs, mask=None):
        return K.not_equal(inputs, '--PAD--')

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.dimensions)


# Function to build model
def build_model(): 
    input_text = Input(shape=(1,), dtype="string")
    embedding = ElmoEmbeddingLayer()(input_text)
    dense = Dense(256, activation="relu", kernel_regularizer=keras.regularizers.l2(0.001))(embedding)  
    pred = Dense(1, activation='sigmoid')(dense)

    model = Model(inputs=[input_text], outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    return model

# Build and fit
model = build_model()
file_path = "elmo_dense_hai.hdf5"
ckpt = ModelCheckpoint(file_path, monitor='val_acc', verbose=1,
                        save_best_only=True, mode='max')
es = EarlyStopping(monitor="val_acc", mode = "max", verbose = 1, patience = 3)
history = model.fit(train_text, 
          train_label,
          validation_data=(test_text, test_label),
          epochs=50,
          batch_size=32, callbacks = [ckpt, es])

with open('trainHistoryDict_elmo_dense.json', 'w') as f:
    json.dump(history.history, f)