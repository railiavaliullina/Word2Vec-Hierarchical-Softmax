import math
import numpy as np


class Word2Vec:

    def __init__(self, hidden_layer_size, vocab_size):
        """
        Class for building Word2Vec model.
        """
        super(Word2Vec, self).__init__()

        self.window = 5
        self.hidden_size = hidden_layer_size
        self.vocab_size = vocab_size

        # scale = 1 / max(1., (2 + 2) / 2.)
        # limit = math.sqrt(3.0 * scale)
        self.w0 = np.random.uniform(-0.5, 0.5, size=(vocab_size, hidden_layer_size)).astype(np.float32) / hidden_layer_size
        self.w1 = np.zeros((vocab_size, hidden_layer_size)).astype(np.float32)


def get_model(hidden_layer_size, vocab_size):
    """
    Gets MLP model.
    :return: MLP model
    """
    np.random.seed(0)
    model = Word2Vec(hidden_layer_size, vocab_size)
    return model
