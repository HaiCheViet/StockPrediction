import functools
import re
import string
from collections import Counter

import nltk
import numpy as np
import pandas as pd
import spacy
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dense, LSTM, Embedding, Input, Concatenate, Lambda
from keras.models import Model

nlp = spacy.load('en', disable=["tagger"])

companies = pd.read_csv(
    'https://datahub.io/core/s-and-p-500-companies-financials/r/constituents-financials.csv')
companies.columns = list(map(lambda x: x.strip().lower(), companies.columns))

companies.index = companies['symbol']
companies = companies[['symbol', 'name', 'sector']]
company_names = companies['name'].values
company_symbols = companies['symbol'].values
company_info = companies[['symbol', 'name', 'name']].values

stop_company_name = ['&', 'the', 'company', 'inc', 'inc.', 'plc',
                     'corp', 'corp.', 'co', 'co.', 'worldwide', 'corporation', 'group', '']
# stop_company_name=[]
splitted_companies = list(map(lambda x: ([x[0]] + [x[1]] + list(filter(
    lambda y: y.lower() not in stop_company_name, x[2].split(' ')))), company_info))
splitted_companies = list(map(lambda x: [x[0]] + [x[1]] + [re.sub(pattern='[^a-zA-Z0-9\s-]',
                                                                  repl='',
                                                                  string=functools.reduce(lambda y, z: y + ' ' + z,
                                                                                          x[2:]))], splitted_companies))


def load_data(file_name, sample_ratio=1, n_class=15, one_hot=True):
    '''load data from .csv file'''
    csv_file = pd.read_csv(file_name)
    x = csv_file["normalize_heading"].values[0:1]
    y = csv_file["price_label"].values[0:1]

    return x, y


def normalize(sent_doc):
    result = []
    for token in sent_doc:
        # Remove punct and stop word
        if not token.is_punct:
            result.append(token.lemma_)
    result = " ".join(result).lower()
    result = re.sub("[^a-z-]", " ", result)
    return re.sub(" +", " ", result)


def tokenize_remove_stopwords_extract_companies_with_spacy(text, sample_date):
    stopwords = nltk.corpus.stopwords.words('english')
    stopwords.append('would')
    stopwords.append('kmh')
    stopwords.append('mph')
    stopwords.append('u')
    stopwords.extend(list(string.ascii_lowercase))

    processed_data = []
    doc = nlp(text)
    sentences = list(doc.sents)

    for sentence in sentences:
        complete_sentence = str(sentence)
        sent_doc = nlp(complete_sentence)
        entities = list(map(str, sent_doc.ents))
        for company in splitted_companies:
            if company[1] in entities or company[2] in complete_sentence or company[0] in entities:
                complete_sentence = normalize(sent_doc)
                processed_data.append((complete_sentence, company[1], sample_date))

    df = pd.DataFrame(processed_data,
                      columns=["complete_sentence", "company", "sample_date"])
    return df


def build_vocabulary(sentence_list):
    unique_words = " ".join(sentence_list).strip().split()
    word_count = Counter(unique_words).most_common()
    vocabulary = {}
    for word, _ in word_count:
        vocabulary[word] = len(vocabulary)

    return vocabulary


# parameter of max word length


def get_voc_vec(document_list, vocabulary):
    time_steps = 100

    """
    # INPUT: raw sentence list
    # OUTPUT: vocabulary vectors list
    :param document_list:
    :param vocabulary:
    :return:
    """
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


# mini-batch generator
def batch_iter(data, labels, batch_size, vocabulary, shuffle=False):
    time_steps = 100
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


def pipeline(x_train, y_train):
    time_steps = 100
    vocabulary = build_vocabulary(x_train)
    # Create datasets (Only take up to time_steps words for memory)
    train_text = x_train.tolist()
    train_text = [' '.join(t.split()[0:time_steps]) for t in train_text]
    train_text = np.array(train_text)
    train_label = y_train

    # mini-batch size
    batch_size = 32

    train_steps, train_batches = batch_iter(train_text,
                                            train_label,
                                            batch_size,
                                            vocabulary)
    return train_batches, train_steps


# embed elmo method
def make_elmo_embedding(x):
    embeddings = elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]

    return embeddings


if __name__ == "__main__":
    elmo_module = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

    x_train, y_train = load_data("train.csv", sample_ratio=1e-2, one_hot=False)

    word_input = Input(shape=(None,), dtype='int32')
    elmo_input = Input(shape=(None,), dtype="string")

    word_embedding = Embedding(input_dim=25399, output_dim=128, mask_zero=True)(word_input)
    elmo_embedding = Lambda(make_elmo_embedding, output_shape=(None, 1024))(elmo_input)

    word_embedding = Concatenate()([word_embedding, elmo_embedding])

    model = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(word_embedding)

    pred = Dense(1, activation='sigmoid')(model)
    model = Model(inputs=[word_input, elmo_input], outputs=pred)

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()

    # In[15]:

    train_batches, train_steps = pipeline(x_train, y_train)
    model.load_weights("elmo_lstm_hai_bare.hdf5")
    result = model.predict_generator(train_batches, train_steps, verbose=1)
    result = np.where(result > 0.5, 1, 0)
    print(result)
    print(y_train)
