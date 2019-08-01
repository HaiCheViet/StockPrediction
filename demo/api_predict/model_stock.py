from collections import defaultdict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from keras.layers import Dense, LSTM, Embedding, Input, Concatenate, Lambda
from keras.models import Model
from api_predict.pipe_line import pipeline, tokenize_remove_stopwords_extract_companies_with_spacy


class ModelStock(object):
    elmo_module = hub.Module("https://tfhub.dev/google/elmo/2", trainable=False)

    def __init__(self, model_path):
        self.sess = tf.Session()
        self.model_path = model_path

        self.model = self.init_model()

    def make_elmo_embedding(self, x):
        embeddings = self.elmo_module(tf.squeeze(tf.cast(x, tf.string)), signature="default", as_dict=True)["elmo"]

        return embeddings

    def parse_data(self, data):
        self.summary = tokenize_remove_stopwords_extract_companies_with_spacy(data["whole_data"], data["date"])

        self.x_train = self.summary["complete_sentence"].values
        self.company = self.summary["company"].values
        if len(self.x_train) < 1:
            return False

        self.x_train = np.insert(self.x_train, 0, np.array("Nan"))
        print(self.x_train)
        self.y_train = np.zeros(self.x_train.shape)

        self.train_batch, self.train_step = pipeline(self.x_train, self.y_train)
        return True

    def predict(self):
        result = defaultdict(int)

        predict = self.model.predict_generator(self.train_batch, self.train_step, verbose=1)
        predict = np.where(predict > 0.5, 1, -1)
        count = 0
        for k, v in zip(self.company, predict):
            if count == 0:
                count += 1
            result[k] += int(v[0])
        return result

    def init_model(self):
        word_input = Input(shape=(None,), dtype='int32')
        elmo_input = Input(shape=(None,), dtype="string")

        word_embedding = Embedding(input_dim=25399, output_dim=128, mask_zero=True)(word_input)
        elmo_embedding = Lambda(self.make_elmo_embedding, output_shape=(None, 1024))(elmo_input)

        word_embedding = Concatenate()([word_embedding, elmo_embedding])

        model = LSTM(100, dropout=0.2, recurrent_dropout=0.2)(word_embedding)

        pred = Dense(1, activation='sigmoid')(model)
        model = Model(inputs=[word_input, elmo_input], outputs=pred)

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())
        model.load_weights(self.model_path)
        print("done loading model")
        return model


if __name__ == "__main__":
    Model = ModelStock("elmo_lstm_hai_bare.hdf5")
    data = {
        "whole_data": " buoyed by optimism about the spending outlook at Wal-Mart Stores Inc.",
        "date": "Now"
    }
    Model.parse_data(data)
    print(Model.predict())
