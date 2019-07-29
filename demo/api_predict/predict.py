import falcon
import tensorflow as tf
import json

graph = tf.get_default_graph()


class PredictResource(object):

    def __init__(self, model):
        self.model = model

    def on_get(self, req, resp):
        resp.status = falcon.HTTP_200
        resp.body = 'Hello World!'

    def on_post(self, req, resp):
        """
        (echo -n '{"image": "'; four_test.png; echo '"}') |
        curl -H "Content-Type: application/json" -d @-  http://127.0.0.1:8000/predict
        """

        body = req.stream.read()

        data = json.loads(body.decode('utf-8'))
        print(data)

        with graph.as_default():
            flag = self.model.parse_data(data)
            if not flag:
                result = {"Company": 0}
            else:
                result = self.model.predict()


        resp.status = falcon.HTTP_200
        resp.body = json.dumps(result)
