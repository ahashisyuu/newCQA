import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer
from BertCQAModel import BertCQAModel as BaseModel


class BertCQAModel(BaseModel):
    def __init__(self, is_training: bool, features: dict, initializer=xavier_initializer(), config=None):
        self.is_training = is_training
        self.initializer = initializer
        self.config = config

        self.input_keep_prob = 0.9
        self.output_keep_prob = 0.9
        self.state_keep_prob = 0.9

        with tf.variable_scope("CQAModel"):
            self.sent1 = features["sent1"]
            self.sent2 = features["sent2"]
            self.sent3 = features["sent3"]

            self.mask1 = features["mask1"]
            self.mask2 = features["mask2"]
            self.mask3 = features["mask3"]

            self.mark0 = features["mark0"]
            self.mark1 = features["mark1"]
            self.mark2 = features["mark2"]
            self.mark3 = features["mark3"]

            self.q_type = features["q_type"]
            self.labels = features["labels"]

            self.output = None

            self.encoding_size = self.sent1.get_shape()[-1]
            self._keys_embedding = None
            self.build_model()

    def build_model(self):
        pass

    def get_output(self):
        return self.mark0










