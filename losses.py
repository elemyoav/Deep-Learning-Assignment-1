import numpy as np
from utils import col_mean, log

class SoftmaxLayer:
    """
    Softmax layer, computes weighted softmax as learned in class.
    uses cross entropy loss as a loss function.
    """

    def __init__(self, input_size, output_size):
        self.W = np.random.randn(input_size, output_size)
        self.input_size = input_size
        self.output_size = output_size

    def forward(self, X):
        """
        computes softmax(X)
        
        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        Z = np.dot(X.T, self.W)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        exp_Z = np.exp(Z)
        return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
    
    def grad_x(self, X, C):
        """
        Computes the gradient of the loss with respect to x
        
        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a vector dx of size (input_size, 1)
        """

        return col_mean(np.dot(self.W, (self.forward(X) - C.T).T))
      
    def grad_w(self, X, C):
        """
        Computes the gradient of the loss with respect to W

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a matrix dW of size (input_size, output_size)
        """

        return np.dot(X, self.forward(X) - C.T)
    
    def grad_b(self, X, C):
        """
        Computes the gradient of the loss with respect to b

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a vector db of size (output_size, 1)
        """
        return None
    
    def loss(self, X, C):
        """
        Computes the cross entropy loss

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        the loss, a scalar
        """

        return np.sum(-log(self.forward(X)) * C.T) / X.shape[1]

    def update_weights(self, Θ, lr):
        """
        recives a vector dΘ = [vec(dW), db].T
        and updates W and b

        Parameters:
        Θ is a vector of size (input_size * output_size + output_size, 1)
        lr is the learning rate

        Returns:
        None
        """

        dW, _ = Θ
        self.W -= lr * dW
    
    def unpack_Θ(self, dΘ):
        """
        recives a vector dΘ = [vec(dW), dx].T
        and unpacks it to dW, dx

        Parameters:
        dΘ is a vector of size (input_size * output_size + input_size, 1)

        Returns:
        dW is a matrix of size (input_size, output_size)
        dx is a vector of size (input_size, 1)
        """

        dW = dΘ[:self.W.size].reshape(self.W.shape)
        dx = dΘ[self.W.size:].reshape(-1, 1)
        return dW, dx
    
    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W.size



class LinearLayer:
    """
    A Simple Linear Layer,
    computes Wx + b
    """

    def __init__(self, input_size, output_size):
        self.W = np.random.randn(output_size, input_size)
        self.b = np.random.randn(output_size, 1)
        self.input_size = input_size
        self.output_size = output_size
    
    def forward(self, x):
        """
        computes Wx + b

        Parameters:
        x is a vector of size (input_size, 1)

        Returns:
        a vector of size (output_size, 1)
        """

        return np.dot(self.W, x) + self.b

    def loss(self, x, y):
        """
        computes the MSE loss

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)
        """

        y_pred = self.forward(x)
        return np.mean((y_pred - y) ** 2)
    
    def update_weights(self, Θ, lr):
        """
        recives a vector dΘ = [vec(dW), db, vec(dx)].T
        and updates W, b and x

        Parameters:
        Θ is a vector of size (input_size * output_size + output_size + input_size, 1)
        lr is the learning rate

        Returns:
        None
        """

        dW, db = Θ
        self.W -= lr * dW
        self.b -= lr * db

    def grad_w(self, x, y):
        """
        Computes the gradient of the loss with respect to W

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a matrix dW of size (output_size, input_size)
        """

        y_pred = self.forward(x)
        return 2 * np.dot((y_pred - y), x.T)

    def grad_b(self, x, y):
        """
        Computes the gradient of the loss with respect to b

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a vector db of size (output_size, 1)
        """

        y_pred = self.forward(x)
        return 2 * np.mean(y_pred - y, axis=1, keepdims=True)

    def grad_x(self, x, y):
        """
        Computes the gradient of the loss with respect to x

        Parameters:
        x is a vector of size (input_size, 1)
        y is a vector of size (output_size, 1)

        Returns:
        a vector dx of size (input_size, 1)
        """

        y_pred = self.forward(x)
        return 2 * np.mean(np.dot(self.W.T, (y_pred - y)), axis=1, keepdims=True)
    
    def unpack_Θ(self, dΘ):
        """
        recives a vector dΘ = [vec(dW), db, vec(dx)].T
        and unpacks it to dW, db, dx

        Parameters:
        dΘ is a vector of size (input_size * output_size + output_size + input_size, 1)

        Returns:
        dW is a matrix of size (output_size, input_size)
        db is a vector of size (output_size, 1)
        dx is a vector of size (input_size, 1)
        """

        dW = dΘ[:self.W.size].reshape(self.W.shape)
        db = dΘ[self.W.size:self.W.size + self.b.size].reshape(self.b.shape)
        dx = dΘ[self.W.size + self.b.size:].reshape(-1, 1)
        return dW, db, dx
    
    def size(self):
        """
        Returns the number of parameters in the layer
        """

        return self.W.size + self.b.size