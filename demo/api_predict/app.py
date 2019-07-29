import os
import falcon
from model_stock import ModelStock
from predict import PredictResource

api = application = falcon.API()


def load_trained_model():
    global model
    model = ModelStock("elmo_lstm_hai_bare.hdf5")
    return model


predict = PredictResource(model=load_trained_model())
api.add_route('/predict', predict)