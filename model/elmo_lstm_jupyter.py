#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import our dependencies
import tensorflow as tf
import pandas as pd
import tensorflow_hub as hub
import os
import re
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.layers import Dense, LSTM, Bidirectional, Embedding, Input, Dropout, TimeDistributed, Concatenate, Lambda, Reshape
from tensorflow.layers import Flatten
from keras.models import Model, load_model
from keras.engine import Layer
import numpy as np
from collections import Counter


# In[2]:


def load_data(file_name, sample_ratio=1, n_class=15, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    x = csv_file["normalize_heading"].values
    y = csv_file["price_label"].values

    return x, y


# In[3]:


x_train, y_train = load_data("train.csv", sample_ratio=1e-2, one_hot=False)
x_test, y_test = load_data("test.csv", one_hot=False)


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


# Get vocabulary vectors from document list
# Vocabulary vector, Unknown word is 1 and padding is 0
# INPUT: raw sentence list
# OUTPUT: vocabulary vectors list
def get_voc_vec(document_list, vocabulary):    
    voc_ind_sentence_list = []
    for document in document_list:
        voc_idx_sentence = []
        word_list = document.split()
        
        for w in range(time_steps):
            if w < len(word_list):
                # pickup vocabulary id and convert unknown word into 1
                voc_idx_sentence.append(vocabulary.get(word_list[w], -1) + 2)
            else:
                # padding with 0
                voc_idx_sentence.append(0)
            
        voc_ind_sentence_list.append(voc_idx_sentence)
        
    return np.array(voc_ind_sentence_list)


vocabulary = build_vocabulary(x_train)


# In[11]:


# Instantiate the elmo model
elmo_module = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

# Initialize session
sess = tf.Session()
K.set_session(sess)

K.set_learning_phase(1)

sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


# In[12]:


# mini-batch generator
def batch_iter(data, labels, batch_size, shuffle=True):
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    print("batch_size", batch_size)
    print("num_batches_per_epoch", num_batches_per_epoch)

    def data_generator():
        data_size = len(data)

        while True:
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
                shuffled_labels = labels[shuffle_indices]
            else:
                shuffled_data = data
                shuffled_labels = labels

            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                                
                X_voc = get_voc_vec(shuffled_data[start_index: end_index], vocabulary)
                                
                sentence_split_list = []
                sentence_split_length_list = []
            
                for sentence in shuffled_data[start_index: end_index]:    
                    sentence_split = sentence.split()
                    sentence_split_length = len(sentence_split)
                    sentence_split += ["NaN"] * (time_steps - sentence_split_length)
                    
                    sentence_split_list.append((" ").join(sentence_split))
                    sentence_split_length_list.append(sentence_split_length)
        
                X_elmo = np.array(sentence_split_list)

                X = [X_voc, X_elmo]
                y = shuffled_labels[start_index: end_index]
                
                yield X, y

    return num_batches_per_epoch, data_generator()


# In[13]:


# embed elmo method
def make_elmo_embedding(x):
    embeddings = elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]
    
    return embeddings


# In[14]:


word_input = Input(shape=(None,), dtype='int32')
elmo_input = Input(shape=(None,), dtype=tf.string)

word_embedding = Embedding(input_dim=len(vocabulary), output_dim=128, mask_zero=True)(word_input)
elmo_embedding = Lambda(make_elmo_embedding, output_shape=(None, 1024))(elmo_input)

word_embedding = Concatenate()([word_embedding, elmo_embedding])

# dense = Dense(256, activation = "relu")(embedding)
# dense = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(word_embedding)
bi_lstm = Bidirectional(LSTM(100, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(word_embedding)
model = TimeDistributed(Dense(100,activation='relu'))(bi_lstm)
# bi_lstm = Flatten()(bi_lstm)
# newdim = tuple([x for x in model.shape.as_list() if x != 1 and x is not None])
pred = Dense(1, activation='sigmoid')(bi_lstm)
pred = pred[0][0][1]
model = Model(inputs=[word_input, elmo_input], outputs=pred)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()


# In[15]:


# Create datasets (Only take up to time_steps words for memory)
train_text = x_train.tolist()
train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
train_text = np.array(train_text)
train_label = y_train

test_text = x_test.tolist()
test_text = [' '.join(t.split()[0:time_steps]) for t in test_text]
test_text = np.array(test_text)
test_label = y_test


# In[16]:


# mini-batch size
batch_size = 32

train_steps, train_batches = batch_iter(train_text,
                                        train_label,
                                        batch_size)
valid_steps, valid_batches = batch_iter(test_text,
                                        test_label,
                                        batch_size)


# In[19]:


logfile_path = './log'
tb_cb = TensorBoard(log_dir=logfile_path, histogram_freq=0)

history = model.fit_generator(train_batches, train_steps,
                              epochs=5, 
                              validation_data=valid_batches,
                              validation_steps=valid_steps,
                              callbacks=[tb_cb])


# In[ ]:




