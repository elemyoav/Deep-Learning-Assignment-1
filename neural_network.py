import numpy as np

class GenericNetwork:
    """
    Generic neural network class, can be used to create any fully connected neural network.

    Parameters:
    output_layer: the output layer of the network, should be either SoftmaxLayer or LinearLayer
    layers: a list of layers, each layer should be an extention of HiddenLayer
    """

    def __init__(self, output_layer, hidden_layers=[]):
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer
        self.cache = [] # cache for forward pass

        self.input_size = hidden_layers[0].input_size if len(hidden_layers) > 0 else output_layer.input_size
        self.output_size = output_layer.output_size

    def loss(self, X, C, clear_cache=True):
        """
        Computes the loss of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)
        clear_cache: if True, clears the cache after computing the loss

        Returns:
        a scalar loss
        """

        self.forward(X)
        loss = self.output_layer.loss(self.cache[-1], C)
        if clear_cache: self.clear_cache()
        return loss
    
    def accuracy(self, X, C):
        """
        Computes the accuracy of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)
        C is a matrix of size (output_size, batch_size)

        Returns:
        a scalar accuracy
        """

        output = self.forward(X)
        self.clear_cache()
        return np.mean(np.argmax(output.T, axis=0) == np.argmax(C, axis=0))
    
    def forward(self, X):
        """
        Computes the output of the network on a given batch of data

        Parameters:neural_network
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        self.cache = [X]
        for layer in self.hidden_layers:
            X = layer.forward(X)
            self.cache.append(X)
        
        Y = self.output_layer.forward(X)
        return Y

    def hidden_layers_forward(self, x):
        """
        Prefroms a forward pass through the hidden layers of the network

        Parameters:
        x is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (o, 1), where o is the size of the output of the last hidden layer
        """

        y = x
        for layer in self.hidden_layers:
            y = layer.forward(y)
        return y
    
    def clear_cache(self):
        """
        Clears the cache of the network
        """

        self.cache = []
    
    def backpropagation(self, X, Y):
        """
        Computes the gradients of the network using backpropagation

        Parameters:
        X is a matrix of size (input_size, batch_size)
        Y is a matrix of size (output_size, batch_size)

        Returns:
        a list of gradients, each gradient is a tuple (dW, db)
        """

        # initialize the list of gradients
        gradients = []

        # preform a forward pass to fill the cache
        self.forward(X)

        # compute the gradient of the loss with respect to the output of the network
        output = self.cache[-1]
        dx = self.output_layer.grad_x(output, Y)
        dW = self.output_layer.grad_w(output, Y)
        db = self.output_layer.grad_b(output, Y)

        # add the gradient of the output layer to the list of gradients
        gradients.append((dW, db))

        # backpropagate the gradient for each layer in reverse order
        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            dW = layer.JacWTMv(self.cache[i], dx)

            db = layer.JacbTMv(self.cache[i], dx)

            dx = layer.JacxTMv(self.cache[i], dx)

            # add the gradient of the layer to the list of gradients
            gradients.append((dW, db))

        # reverse the list of gradients and return it
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        """
        Updates the parameters of each layer of the network using the learning rate

        Parameters:
        gradients: a list of gradients, each gradient is a tuple (dW, db), corresponding to a layer
        lr: the learning rate

        Returns:
        None
        """

        Θ, gradients = gradients[-1], gradients[:-1]
        self.output_layer.update_weights(Θ, lr)

        for i, Θ in enumerate(gradients):
            self.hidden_layers[i].update_weights(Θ, lr)
    
    def size(self):
        """
        Computes the total number of parameters in the network

        Returns:
        an integer
        """

        return sum([layer.size() for layer in self.hidden_layers]) + self.output_layer.size()
    
    def grad_x(self, x, y):
        """
        Computes the gradient of the loss with respect to the input of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for i, layer in reversed(list(enumerate(self.hidden_layers))):
            grad = layer.JacxTMv(cache[i], grad)
        
        self.clear_cache()
        return grad
    
    def grad_W_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the weights of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)
        i is an integer

        Returns:
        The gradient of the loss with respect to the weights of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        if i == len(self.hidden_layers):
            return self.output_layer.grad_w(cache[-1], y)
        
        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.hidden_layers))):
            if j == i:
                grad = layer.JacWTMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad
    
    def grad_b_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the biases of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)
        i is an integer

        Returns:
        The gradient of the loss with respect to the biases of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        if i == len(self.hidden_layers):
            return self.output_layer.grad_b(cache[-1], y)
        
        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.hidden_layers))):
            if j == i:
                grad = layer.JacbTMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad
    
    def JacxMv(self, x, v):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to its input and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """
        self.forward(x)
        cache = self.cache[:-1]
        Jv = v

        for i, layer in list(enumerate(self.hidden_layers)):
            Jv = layer.JacxMv(cache[i], Jv)
        
        self.clear_cache()
        return Jv
    
    def JacWiMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the weights of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.hidden_layers[i].JacWMv(cache[i], v)

        for j, layer in list(enumerate(self.hidden_layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv
    
    def JacbiMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the biases of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.hidden_layers[i].JacbMv(cache[i], v)

        for j, layer in list(enumerate(self.hidden_layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv


class ResidualNeuralNetwork:
    """
    Generic neural network class, can be used to create any fully connected neural network.
    """

    def __init__(self, output_layer, layers=[]):
        self.layers = layers
        self.output_layer = output_layer
        self.cache = []
        self.input_size = layers[0].input_size if len(layers) > 0 else output_layer.input_size
        self.output_size = output_layer.output_size

    def loss(self, X, C, clear_cache=True):
        """
        Computes the loss of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a scalar loss
        """

        self.forward(X)
        loss = self.output_layer.loss(self.cache[-1], C)
        if clear_cache: self.clear_cache()
        return loss
    
    def accuracy(self, X, C):
        """
        Computes the accuracy of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a scalar accuracy
        """

        output = self.forward(X)
        self.clear_cache()
        return np.mean(np.argmax(output.T, axis=0) == np.argmax(C, axis=0))
    
    def forward(self, X):
        """
        Computes the output of the network on a given batch of data

        Parameters:
        X is a matrix of size (input_size, batch_size)

        Returns:
        a matrix of size (output_size, batch_size)
        """

        self.cache = [X]
        for layer in self.layers:
            X = layer.forward(X)
            self.cache.append(X)
        
        Y = self.output_layer.forward(X)
        return Y

    def hidden_layers_forward(self, x):
        """
        Prefroms a forward pass through the hidden layers of the network

        Parameters:
        x is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (o, 1), where o is the size of the output of the last hidden layer
        """

        y = x
        for layer in self.layers:
            y = layer.forward(y)
        return y
    
    def clear_cache(self):
        """
        Clears the cache of the network
        """

        self.cache = []
    
    def backpropagation(self, X, Y):
        """
        Computes the gradients of the network using backpropagation

        Parameters:
        X is a matrix of size (input_size, batch_size)
        Y is a matrix of size (output_size, batch_size)

        Returns:
        a list of gradients, each gradient is a tuple (dW, db)
        """

        # initialize the list of gradients
        gradients = []

        # preform a forward pass to fill the cache
        self.forward(X)
        output = self.cache[-1]

        # compute the gradient of the loss with respect to the output of the network
        dx = self.output_layer.grad_x(output, Y)
        dW = self.output_layer.grad_w(output, Y)
        db = self.output_layer.grad_b(output, Y)

        # add the gradient of the output layer to the list of gradients
        gradients.append((dW, db))

        # backpropagate the gradient for each layer in reverse order
        for i, layer in reversed(list(enumerate(self.layers))):
            dW1 = layer.JacW1TMv(self.cache[i], dx)

            dW2 = layer.JacW2TMv(self.cache[i], dx)

            db1 = layer.Jacb1TMv(self.cache[i], dx)
            
            db2 = layer.Jacb2TMv(self.cache[i], dx)

            dx = layer.JacxTMv(self.cache[i], dx)

            gradients.append((dW1, dW2, db1, db2))
        
        # reverse the list of gradients and return it
        gradients.reverse()
        return gradients

    def update_parameters(self, gradients, lr):
        """
        Updates the parameters of each layer of the network using the learning rate

        Parameters:
        gradients: a list of gradients, each gradient is a tuple (dW, db), corresponding to a layer
        lr: the learning rate

        Returns:
        None
        """

        Θ, gradients = gradients[-1], gradients[:-1]
        self.output_layer.update_weights(Θ, lr)

        for i, Θ in enumerate(gradients):
            self.layers[i].update_weights(Θ, lr)

    def grad_x(self, x, y):
        """
        Computes the gradient of the loss with respect to the input of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for i, layer in reversed(list(enumerate(self.layers))):
            grad = layer.JacxTMv(cache[i], grad)
        
        self.clear_cache()
        return grad
    
    def grad_W1_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the weights of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)
        i is an integer

        Returns:
        The gradient of the loss with respect to the weights of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.layers))):
            if j == i:
                grad = layer.JacW1TMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad
    
    def grad_W2_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the weights of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)

        Returns:
        The gradient of the loss with respect to the weights of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.layers))):
            if j == i:
                grad = layer.JacW2TMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad

    def grad_b1_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the biases of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)
        i is an integer

        Returns:
        The gradient of the loss with respect to the biases of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.layers))):
            if j == i:
                grad = layer.Jacb1TMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad

    def grad_b2_i(self, x, y, i):
        """
        Computes the gradient of the loss with respect to the biases of the i-th layer of the network

        Parameters:
        x is a matrix of size (input_size, 1)
        y is a matrix of size (output_size, 1)

        Returns:
        The gradient of the loss with respect to the biases of the i-th layer
        """

        self.forward(x)
        cache = self.cache

        grad = self.output_layer.grad_x(cache[-1], y)

        for j, layer in reversed(list(enumerate(self.layers))):
            if j == i:
                grad = layer.Jacb2TMv(cache[j], grad)
                break

            grad = layer.JacxTMv(cache[j], grad)
        
        self.clear_cache()
        return grad
    
    def JacxMv(self, x, v):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to its input and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = v

        for i, layer in list(enumerate(self.layers)):
            Jv = layer.JacxMv(cache[i], Jv)
        
        self.clear_cache()
        return Jv

    def JacW1iMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the weights of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.layers[i].JacW1Mv(cache[i], v)

        for j, layer in list(enumerate(self.layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv

    def JacW2iMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the weights of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.layers[i].JacW2Mv(cache[i], v)

        for j, layer in list(enumerate(self.layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv
    
    def Jacb1iMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the biases of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """

        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.layers[i].Jacb1Mv(cache[i], v)

        for j, layer in list(enumerate(self.layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv
    
    def Jacb2iMv(self, x, v, i):
        """
        Computes the product of the Jacobian of the output of hidden_layers_forward with respect to the biases of the i-th layer and a vector

        Parameters:
        x is a matrix of size (input_size, 1)
        v is a matrix of size (input_size, 1)

        Returns:
        a matrix of size (input_size, 1)
        """
        
        self.forward(x)
        cache = self.cache[:-1]
        Jv = self.layers[i].Jacb2Mv(cache[i], v)

        for j, layer in list(enumerate(self.layers[i+1:])):
            Jv = layer.JacxMv(cache[i+1+j], Jv)
        
        self.clear_cache()
        return Jv

