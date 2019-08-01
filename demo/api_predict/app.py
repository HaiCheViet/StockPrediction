import os
import falcon
from api_predict.model_stock import ModelStock
from api_predict.predict import PredictResource

api = application = falcon.API()


def load_trained_model():
    global model
    model = ModelStock("api_predict/elmo_lstm_hai_bare.hdf5")
    return model


predict = PredictResource(model=load_trained_model())
api.add_route('/predict', predict)
