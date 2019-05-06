import numpy as np


class SGD:
    """
    Implements vanilla SGD update
    """

    def update(self, w, d_w, learning_rate):
        """
        Performs SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        return w - d_w * learning_rate


class MomentumSGD:
    """
    Implements Momentum SGD update
    """

    def __init__(self, momentum=0.9):
        self.momentum = 0.9
        self.velocity = None

    def update(self, w, d_w, learning_rate):
        """
        Performs Momentum SGD update

        Arguments:
        w, np array - weights
        d_w, np array, same shape as w - gradient
        learning_rate, float - learning rate

        Returns:
        updated_weights, np array same shape as w
        """
        # Implement momentum update
        # Hint: you'll need to introduce some variables to remember
        # velocity from the previous updates
        if self.velocity is None:
            self.velocity = - learning_rate*d_w
        else:
            self.velocity = self.velocity*self.momentum - learning_rate*d_w
        return w + self.velocity
