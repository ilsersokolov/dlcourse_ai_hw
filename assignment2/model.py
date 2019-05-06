import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization, softmax


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # Create necessary layers
        self.layers = [
            FullyConnectedLayer(n_input, hidden_layer_size),
            ReLULayer(),
            FullyConnectedLayer(hidden_layer_size, n_output),
            # ReLULayer()
        ]

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for param_name, param in self.params().items():
            param.grad = np.zeros_like(param.value)

        # Compute loss and fill param gradients
        # by running forward and backward passes through the model
        result = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            result = layer.forward(result)
        loss, grad = softmax_with_cross_entropy(result, y)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for param_name, param in self.params().items():
            loss_r, grad = l2_regularization(param.value, self.reg)
            param.grad += grad
            loss += loss_r

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)

        result = self.layers[0].forward(X)
        for layer in self.layers[1:]:
            result = layer.forward(result)
        pred = np.argmax(softmax(result), axis=1)
        return pred

    def params(self):
        result = {}

        # Implement aggregating all of the params
        i = 0
        for layer in self.layers:
            param = layer.params()
            if param:
                result['W'+str(i)] = param['W']
                result['B'+str(i)] = param['B']
                i += 1

        return result
