import numpy as np
import utils

class Model(object):
    def __init__(self, num_features, num_classes, theta=0.1, bias=True, model_path='Model.pkl'):
        """
        initialize the model
        :param num_features: number of features for each row data
        :param num_classes: total number of classes for the target
        :param theta: multiplier of weight random initialization
        :param bias: bias term
        :param model_path: the path to save trained model file
        """
        self.num_features = num_features
        self.num_classes = num_classes
        self.bias = bias
        self.model_path = model_path
        if self.bias is False:
            self.weights = np.random.randn(self.num_features, self.num_classes) * theta
        else:
            self.weights = np.random.randn(self.num_features + 1, self.num_classes) * theta

    def predict(self, inp):
        # inp is the input vector which has the same dim as the features
        # inp has the shape (number_of_records, feature_dim)
        # append one column if add bias term
        if self.bias is True:
            inp = np.append(inp, np.ones((inp.shape[0], 1)), axis=1)
        output = utils.softmax(np.matmul(inp, self.weights), axis=1)
        return output

    def parameters(self, flatten=False):
        if flatten is False:
            return self.weights
        else:
            return np.ndarray.flatten(self.weights)