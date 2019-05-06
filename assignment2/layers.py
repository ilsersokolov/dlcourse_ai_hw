import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # Copy from the previous assignment
    loss = reg_strength*np.sum(W*W)
    grad = 2*reg_strength*W
    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # implement softmax
    # Your final implementation shouldn't have any loops
    # print(predictions.shape)
    predictions_ = predictions.copy()
    if predictions.shape[0] == predictions.size:
        predictions_ -= np.max(predictions_)
        e = np.exp(predictions_)
        probs = e/e.sum()
    else:
        predictions_ -= np.max(predictions_, axis=1)[:, np.newaxis]
        e = np.exp(predictions_)
        probs = e/e.sum(axis=1)[:, np.newaxis]
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss: single value
    '''
    # implement cross-entropy
    # Your final implementation shouldn't have any loops
    if probs.shape[0] == probs.size:
        loss = -np.log(probs[target_index])[0]
    else:
        loss_arr = - \
            np.log(probs[np.arange(probs.shape[0]), target_index.reshape(-1)])
        loss = np.mean(loss_arr)
    return loss


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # Copy from the previous assignment
    probs = softmax(predictions)
    loss = cross_entropy_loss(probs, target_index)
    dprediction = probs.copy()
    if dprediction.shape[0] == dprediction.size:
        dprediction[target_index] -= 1
    else:
        dprediction[np.arange(dprediction.shape[0]),
                    target_index.reshape(-1)] -= 1
        dprediction /= dprediction.shape[0]

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        self.X = None

    def forward(self, X):
        # Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        result = X.copy()
        result[X < 0] = 0
        self.X = result.copy()
        return result

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # Implement backward pass
        # Your final implementation shouldn't have any loops
        dX = self.X
        dX[dX > 0] = 1
        d_result = dX*d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        d_result = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
