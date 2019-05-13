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
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # copy from the previous assignment
        result = X.copy()
        result[X < 0] = 0
        self.X = result.copy()
        return result

    def backward(self, d_out):
        # copy from the previous assignment
        dX = self.X
        dX[dX > 0] = 1
        d_result = dX*d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # copy from the previous assignment
        self.X = X
        result = np.dot(X, self.W.value) + self.B.value
        return result

    def backward(self, d_out):
        # copy from the previous assignment

        d_result = np.dot(d_out, self.W.value.T)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}


class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer

        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        if self.padding > 0:
            self.X = np.zeros(
                (batch_size, height+2*self.padding, width+2*self.padding, channels))
            self.X[:, self.padding:-self.padding,
                   self.padding:-self.padding, :] = X
            batch_size, height, width, channels = self.X.shape
        else:
            self.X = X.copy()
        filter_size, filter_size, in_channels, out_channels = self.W.value.shape
        out_height = height - filter_size + 1
        out_width = width - filter_size + 1
        W = self.W.value.reshape(
            filter_size*filter_size*in_channels, out_channels)
        result = np.zeros((batch_size, out_height, out_width, out_channels))

        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # Implement forward pass for specific location
                X_ = self.X[:, y:y+filter_size, x:x+filter_size,
                            :].reshape(batch_size, filter_size*filter_size*channels)
                result[:, y, x, :] = np.dot(X_, W) + self.B.value

        return result

    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, in_channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape

        # Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        filter_size = height - out_height + 1
        W = self.W.value.reshape(
            filter_size*filter_size*in_channels, out_channels)
        dW = np.zeros_like(self.W.grad)
        dB = np.zeros_like(self.B.grad)

        # Try to avoid having any other loops here too
        d_result = np.zeros_like(self.X)
        print("Here")
        for y in range(out_height):
            for x in range(out_width):
                # Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                X_ = self.X[:, y:y+filter_size, x:x+filter_size,:]
                X_ = X_.reshape(batch_size, filter_size*filter_size*in_channels)
                d_out_ = d_out[:, y, x, :]
                d_out_ = d_out_.reshape(batch_size, out_channels)
                grad = np.dot(d_out_, W.T)
                grad = grad.reshape(batch_size, filter_size, filter_size, in_channels)
                d_result[:, y:y+filter_size, x:x+filter_size, :] += grad
                # dW_ = np.dot(X_.T, d_out_)
                # dW_ = dW_.reshape(filter_size,filter_size, in_channels, out_channels)
                # dW += dW_
                # dB += np.sum(d_out_, axis=0)

        if self.padding > 0:
            d_result = d_result[:, self.padding:-self.padding, self.padding:-self.padding, :]

        n = out_height*out_width
        self.W.grad += dW/n
        self.B.grad += dB/n

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        if pool_size < stride:
            ValueError('pool size mast be greater or eq stride')
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        # Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        self.X = X.copy()
        out_height = int((height - self.pool_size)/self.stride + 1)
        out_width = int((width - self.pool_size)/self.stride + 1)
        if height != self.pool_size + (out_height - 1) * self.stride or width != self.pool_size + (out_width - 1) * self.stride:
            ValueError('wrong pool_size or stride')
        result = np.zeros((batch_size, out_height, out_width, channels))
        self.grad = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                for ch in range(channels):
                    for bs in range(batch_size):
                        X_ = self.X[bs, y * self.stride:y * self.stride + self.pool_size,
                                    x * self.stride:x * self.stride + self.pool_size, ch]
                        result[bs, y, x, ch] = np.max(X_)
        return result

    def backward(self, d_out):
        # Implement maxpool backward pass
        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        d_result = np.zeros_like(self.X)
        for y in range(out_height):
            for x in range(out_width):
                for ch in range(channels):
                    for bs in range(batch_size):
                        X_ = self.X[bs, y * self.stride:y * self.stride + self.pool_size,
                                    x * self.stride:x * self.stride + self.pool_size, ch]
                        g_y,g_x=np.unravel_index(np.argmax(X_, axis=None), X_.shape)
                        d_result[bs,g_y,g_x,ch] = d_out[bs,y,x,ch]

        return d_result

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, height*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height*width*channels)

    def backward(self, d_out):
        # Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}
