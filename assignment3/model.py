import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization, softmax
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # Create necessary layers
        self.layers = [
            ConvolutionalLayer(input_shape[2],conv1_channels,3,3),
            ReLULayer(),
            MaxPoolingLayer(4,4),
            ConvolutionalLayer(conv1_channels,conv2_channels,3,3),
            ReLULayer(),
            MaxPoolingLayer(4,4),
            Flattener(),
            FullyConnectedLayer(3*3*conv2_channels, n_output_classes),
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param_name, param in self.params().items():
            param.grad = np.zeros_like(param.value)
        # Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        result = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            result = layer.forward(result)
        loss, grad = softmax_with_cross_entropy(result, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        return loss

    def predict(self, X):
        # You can probably copy the code from previous assignment
        pred = np.zeros(X.shape[0], np.int)

        result = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            result = layer.forward(result)
        pred = np.argmax(softmax(result), axis=1)
        return pred

    def params(self):
        result = {}

        # Aggregate all the params from all the layers
        # which have parameters
        i = 0
        for layer in self.layers:
            param = layer.params()
            if param:
                result['W'+str(i)] = param['W']
                result['B'+str(i)] = param['B']
                i += 1

        return result
