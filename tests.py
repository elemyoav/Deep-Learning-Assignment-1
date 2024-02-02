import numpy as np
import matplotlib.pyplot as plt
from utils import vectorize

def output_layer_grad_x_test(output_layer):
    """
    Preforms the gradient test on an output layer (in our case, softmax layer or linear layer)
    with respect to x, and plots the results.

    Parameters:
    output_layer: an output layer object

    Returns:
    None
    """

    # create random x, y, dx
    x = np.random.randn(output_layer.input_size, 1)
    y = np.zeros((output_layer.output_size, 1))
    index = np.random.randint(0, output_layer.output_size)
    y[index] = 1
    dx = np.random.randn(output_layer.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    grad_x = output_layer.grad_x(x, y)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        f1 = output_layer.loss(x + eps * dx, y)
        f2 = output_layer.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(dx.T, grad_x)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for output layer with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def output_layer_grad_W_test(output_layer):
    """
    Preforms the gradient test on an output layer (in our case, softmax layer or linear layer)
    with respect to W, and plots the results.

    Parameters:
    output_layer: an output layer object

    Returns:
    None
    """

    # create random x, y, dW
    x = np.random.randn(output_layer.input_size, 1)
    y = np.zeros((output_layer.output_size, 1))
    index = np.random.randint(0, output_layer.output_size)
    y[index] = 1
    dW = np.random.randn(*output_layer.W.shape)
    dW = dW / np.linalg.norm(dW)

    grad_W = output_layer.grad_w(x, y)

    # vectorize dW and grad_W, so we can use them in the second order error
    vec_dW = vectorize(dW)
    vec_grad_W = vectorize(grad_W)


    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):

        output_layer.W += eps * dW
        f1 = output_layer.loss(x, y)
        output_layer.W -= eps * dW

        f2 = output_layer.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(vec_dW.T, vec_grad_W)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for output layer with respect to W')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def hidden_layer_Jac_x_test(hidden_layer):
    """
    Preforms the Jacobian test on a hidden layer (in our case, tanh layer or relu layer)
    with respect to x, and plots the results.

    Parameters:
    hidden_layer: a hidden layer object

    Returns:
    None
    """

    # create random x, dx
    x = np.random.randn(hidden_layer.input_size, 1)
    dx = np.random.randn(hidden_layer.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        f1 = hidden_layer.forward(x + eps * dx)
        f2 = hidden_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * hidden_layer.JacxMv(x, dx)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for hidden layer with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def hidden_layer_Jac_W_test(hidden_layer):
    """
    Preforms the Jacobian test on a hidden layer (in our case, tanh layer or relu layer)
    with respect to W, and plots the results.

    Parameters:
    hidden_layer: a hidden layer object

    Returns:
    None
    """

    # create random x, dW
    x = np.random.randn(hidden_layer.input_size, 1)
    dW = np.random.randn(hidden_layer.output_size, hidden_layer.input_size)
    dW = dW / np.linalg.norm(dW)

    # vectorize dW, so we can use it in the second order error
    vec_dW = vectorize(dW)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        hidden_layer.W += eps * dW
        f1 = hidden_layer.forward(x)
        hidden_layer.W -= eps * dW
        f2 = hidden_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * hidden_layer.JacWMv(x, vec_dW)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Jacobian test for hidden layer with respect to W')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def hidden_layer_Jac_b_test(hidden_layer):
    """
    Preforms the Jacobian test on a hidden layer (in our case, tanh layer or relu layer)
    with respect to b, and plots the results.

    Parameters:
    hidden_layer: a hidden layer object

    Returns:
    None
    """

    # create random x, db
    x = np.random.randn(hidden_layer.input_size, 1)
    db = np.random.randn(hidden_layer.output_size, 1)
    db = db / np.linalg.norm(db)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        hidden_layer.b += eps * db
        f1 = hidden_layer.forward(x)
        hidden_layer.b -= eps * db

        f2 = hidden_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * hidden_layer.JacbMv(x, db)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for hidden layer with respect to b')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def network_grad_x_test(network):
    """
    Preforms the gradient test on a network with respect to x, and plots the results.

    Parameters:
    network: a network object

    Returns:
    None
    """

    # create random x, y, dx
    x = np.random.randn(network.input_size, 1)
    y = np.zeros((network.output_size, 1))
    index = np.random.randint(0, network.output_size)
    y[index] = 1
    dx = np.random.randn(network.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    grad_x = network.grad_x(x, y)

    eps = 1.
    
    E1 = []
    E2 = []

    for _ in range(20):
        f1 = network.loss(x + eps * dx, y)
        f2 = network.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(dx.T, grad_x)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Gradient test for network with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def network_grad_Wi_test(network, i):

    """
    Preforms the gradient test on a network with respect to W of the ith hidden layer, and plots the results.

    Parameters:
    network: a network object

    Returns:
    None
    """


    # get the ith hidden layer's W
    if i == len(network.hidden_layers):
        W = network.output_layer.W
    else:
        W = network.hidden_layers[i].W

    # create random x, y, dW
    x = np.random.randn(network.input_size, 1)
    y = np.zeros((network.output_size, 1))
    index = np.random.randint(0, network.output_size)
    y[index] = 1
    dW = np.random.randn(*W.shape)
    dW = dW / np.linalg.norm(dW)

    # vectorize dW, so we can use it in the second order error
    vec_dW = vectorize(dW)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        W += eps * dW
        f1 = network.loss(x, y)
        W -= eps * dW
        f2 = network.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(vec_dW.T, vectorize(network.grad_W_i(x, y, i)))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for network with respect to W of the {}th hidden layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def network_grad_bi_test(network, i):
    
    """
    Preforms the gradient test on a network with respect to b of the ith hidden layer, and plots the results.

    Parameters:
    network: a network object

    Returns:
    None
    """


    # get the ith hidden layer's b
    if i == len(network.hidden_layers):
        b = network.output_layer.b
    else:
        b = network.hidden_layers[i].b

    # create random x, y, db
    x = np.random.randn(network.input_size, 1)
    y = np.zeros((network.output_size, 1))
    index = np.random.randint(0, network.output_size)
    y[index] = 1
    db = np.random.randn(*b.shape)
    db = db / np.linalg.norm(db)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        b += eps * db
        f1 = network.loss(x, y)
        b -= eps * db
        f2 = network.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(db.T, network.grad_b_i(x, y, i))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for network with respect to b of the {}th hidden layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def network_JacxMv_test(network):
    """
    Preforms the Jacobian test on a network with respect to x, and plots the results.

    Parameters:
    network: a network object

    Returns:
    None
    """
    
    # create random x, dx
    x = np.random.randn(network.input_size, 1)
    dx = np.random.randn(network.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    Jv = network.JacxMv(x, dx)

    eps = 1.

    E1 = []
    E2 = []
    
    for _ in range(30):
        f1 = network.hidden_layers_forward(x + eps * dx)
        f2 = network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * Jv))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for network with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def network_JacWiMv_test(network, i):
    """
    Preforms the Jacobian test on a network with respect to W of the ith hidden layer, and plots the results.

    Parameters:
    network: a network object
    i: the index of the hidden layer

    Returns:
    None
    """

    x = np.random.randn(network.input_size, 1)
    dWi = np.random.randn(*network.hidden_layers[i].W.shape)
    dWi = dWi / np.linalg.norm(dWi)
    vec_dWi = vectorize(dWi)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        network.hidden_layers[i].W += eps * dWi
        f1 = network.hidden_layers_forward(x)
        network.hidden_layers[i].W -= eps * dWi
        f2 = network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * network.JacWiMv(x, vec_dWi, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for network with respect to W of the {}th hidden layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def network_JacbiMv_test(network, i):
    """
    Preforms the Jacobian test on a network with respect to b of the ith hidden layer, and plots the results.

    Parameters:
    network: a network object
    i: the index of the hidden layer

    Returns:
    None
    """

    x = np.random.randn(network.input_size, 1)
    dbi = np.random.randn(*network.hidden_layers[i].b.shape)
    dbi = dbi / np.linalg.norm(dbi)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        network.hidden_layers[i].b += eps * dbi
        f1 = network.hidden_layers_forward(x)
        network.hidden_layers[i].b -= eps * dbi
        f2 = network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * network.JacbiMv(x, dbi, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for network with respect to b of the {}th hidden layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def residual_layer_Jac_x_test(residual_layer):

    """
    Preforms the Jacobian test on a residual layer with respect to x, and plots the results.

    Parameters:
    residual_layer: a residual layer object

    Returns:
    None
    """

    # create random x, dx
    x = np.random.randn(residual_layer.input_size, 1)
    dx = np.random.randn(residual_layer.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        f1 = residual_layer.forward(x + eps * dx)
        f2 = residual_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_layer.JacxMv(x, dx)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual layer with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def residual_layer_Jac_W1_test(residual_layer):
    """
    Preforms the Jacobian test on a residual layer with respect to W1, and plots the results.

    Parameters:
    residual_layer: a residual layer object

    Returns:
    None
    """

    # create random x, dW1
    x = np.random.randn(residual_layer.input_size, 1)
    dW1 = np.random.randn(residual_layer.output_size, residual_layer.input_size)
    dW1 = dW1 / np.linalg.norm(dW1)

    # vectorize dW1, so we can use it in the second order error
    vec_dW1 = vectorize(dW1)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_layer.W1 += eps * dW1
        f1 = residual_layer.forward(x)
        residual_layer.W1 -= eps * dW1
        f2 = residual_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_layer.JacW1Mv(x, vec_dW1)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Jacobian test for residual layer with respect to W1')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def residual_layer_Jac_W2_test(residual_layer):
    """
    Preforms the Jacobian test on a residual layer with respect to W2, and plots the results.

    Parameters:
    residual_layer: a residual layer object

    Returns:
    None
    """

    # create random x, dW2
    x = np.random.randn(residual_layer.input_size, 1)
    dW2 = np.random.randn(residual_layer.input_size, residual_layer.output_size)
    dW2 = dW2 / np.linalg.norm(dW2)

    # vectorize dW2, so we can use it in the second order error
    vec_dW2 = vectorize(dW2)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_layer.W2 += eps * dW2
        f1 = residual_layer.forward(x)
        residual_layer.W2 -= eps * dW2
        f2 = residual_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_layer.JacW2Mv(x, vec_dW2)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Jacobian test for residual layer with respect to W2')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_layer_Jac_b1_test(residual_layer):
    """
    Preforms the Jacobian test on a residual layer with respect to b1, and plots the results.

    Parameters:
    residual_layer: a residual layer object

    Returns:
    None
    """

    # create random x, db1
    x = np.random.randn(residual_layer.input_size, 1)
    db1 = np.random.randn(residual_layer.output_size, 1)
    db1 = db1 / np.linalg.norm(db1)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_layer.b1 += eps * db1
        f1 = residual_layer.forward(x)
        residual_layer.b1 -= eps * db1
        f2 = residual_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_layer.Jacb1Mv(x, db1)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual layer with respect to b1')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.legend()
    plt.yscale('log')
    plt.show()

def residual_layer_Jac_b2_test(residual_layer):
        
    """
    Preforms the Jacobian test on a residual layer with respect to b2, and plots the results.

    Parameters:
    residual_layer: a residual layer object

    Returns:
    None
    """

    # create random x, db2
    x = np.random.randn(residual_layer.input_size, 1)
    db2 = np.random.randn(residual_layer.input_size, 1)
    db2 = db2 / np.linalg.norm(db2)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_layer.b2 += eps * db2
        f1 = residual_layer.forward(x)
        residual_layer.b2 -= eps * db2
        f2 = residual_layer.forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_layer.Jacb2Mv(x, db2)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual layer with respect to b2')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()



def residual_network_grad_x_test(resnet): 
    """
    Preforms the gradient test on a residual network with respect to x, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    # create random x, y, dx
    x = np.random.randn(resnet.input_size, 1)
    y = np.zeros((resnet.output_size, 1))
    index = np.random.randint(0, resnet.output_size)
    y[index] = 1
    dx = np.random.randn(resnet.input_size, 1)
    dx = dx / np.linalg.norm(dx)

    grad_x = resnet.grad_x(x, y)
    eps = 1.
    
    E1 = []
    E2 = []

    for _ in range(20):
        f1 = resnet.loss(x + eps * dx, y)
        f2 = resnet.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(dx.T, grad_x)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Gradient test for residual network with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.legend()
    plt.yscale('log')
    plt.show()


def residual_network_grad_W1i_test(resnet, i):
    """
    Preforms the gradient test on a residual network with respect to W1 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    # create random x, y, dW1
    x = np.random.randn(resnet.input_size, 1)
    y = np.zeros((resnet.output_size, 1))
    index = np.random.randint(0, resnet.output_size)
    y[index] = 1
    dW1i = np.random.randn(*resnet.layers[i].W1.shape)
    dW1i = dW1i / np.linalg.norm(dW1i)
    vec_dW1i = vectorize(dW1i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        resnet.layers[i].W1 += eps * dW1i
        f1 = resnet.loss(x, y)
        resnet.layers[i].W1 -= eps * dW1i
        f2 = resnet.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(vec_dW1i.T, vectorize(resnet.grad_W1_i(x, y, i)))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for residual network with respect to W1 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_network_grad_W2i_test(resnet, i):
    """
    Preforms the gradient test on a residual network with respect to W2 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    # create random x, y, dW2
    x = np.random.randn(resnet.input_size, 1)
    y = np.zeros((resnet.output_size, 1))
    index = np.random.randint(0, resnet.output_size)
    y[index] = 1
    dW2i = np.random.randn(*resnet.layers[i].W2.shape)
    dW2i = dW2i / np.linalg.norm(dW2i)

    # vectorize dW2, so we can use it in the second order error
    vec_dW2i = vectorize(dW2i)

    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        resnet.layers[i].W2 += eps * dW2i
        f1 = resnet.loss(x, y)
        resnet.layers[i].W2 -= eps * dW2i
        f2 = resnet.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(vec_dW2i.T, vectorize(resnet.grad_W2_i(x, y, i)))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for residual network with respect to W2 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()  



def residual_network_grad_b1i_test(resnet, i):
    """
    Preforms the gradient test on a residual network with respect to b1 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    # create random x, y, db1
    x = np.random.randn(resnet.input_size, 1)
    y = np.zeros((resnet.output_size, 1))
    index = np.random.randint(0, resnet.output_size)
    y[index] = 1
    db1i = np.random.randn(*resnet.layers[i].b1.shape)
    db1i = db1i / np.linalg.norm(db1i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        resnet.layers[i].b1 += eps * db1i
        f1 = resnet.loss(x, y)
        resnet.layers[i].b1 -= eps * db1i
        f2 = resnet.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(db1i.T, resnet.grad_b1_i(x, y, i))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5

    plt.title('Gradient test for residual network with respect to b1 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_network_grad_b2i_test(resnet, i):
    """
    Preforms the gradient test on a residual network with respect to b2 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    # create random x, y, db2   
    x = np.random.randn(resnet.input_size, 1)
    y = np.zeros((resnet.output_size, 1))
    index = np.random.randint(0, resnet.output_size)
    y[index] = 1
    db2i = np.random.randn(*resnet.layers[i].b2.shape)
    db2i = db2i / np.linalg.norm(db2i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        resnet.layers[i].b2 += eps * db2i
        f1 = resnet.loss(x, y)
        resnet.layers[i].b2 -= eps * db2i
        f2 = resnet.loss(x, y)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * np.dot(db2i.T, resnet.grad_b2_i(x, y, i))))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Gradient test for residual network with respect to b2 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()  


def residual_network_JacxMv_test(residual_network):
    
    x = np.random.randn(residual_network.input_size, 1)
    dx = np.random.randn(residual_network.input_size, 1)
    dx = dx / np.linalg.norm(dx)
    Jv = residual_network.JacxMv(x, dx)

    eps = 1.

    E1 = []
    E2 = []
    
    for _ in range(30):
        f1 = residual_network.hidden_layers_forward(x + eps * dx)
        f2 = residual_network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * Jv))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual network with respect to x')
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_network_JacW1Mv_test(residual_network, i):
    """
    Preforms the Jacobian test on a residual network with respect to W1 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    x = np.random.randn(residual_network.input_size, 1)
    dW1i = np.random.randn(*residual_network.layers[i].W1.shape)
    dW1i = dW1i / np.linalg.norm(dW1i)
    vec_dW1i = vectorize(dW1i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_network.layers[i].W1 += eps * dW1i
        f1 = residual_network.hidden_layers_forward(x)
        residual_network.layers[i].W1 -= eps * dW1i
        f2 = residual_network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_network.JacW1iMv(x, vec_dW1i, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual network with respect to W1 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def residual_network_JacW2Mv_test(residual_network, i):
    """
    Preforms the Jacobian test on a residual network with respect to W2 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """
    x = np.random.randn(residual_network.input_size, 1)
    dW2i = np.random.randn(*residual_network.layers[i].W2.shape)
    dW2i = dW2i / np.linalg.norm(dW2i)
    vec_dW2i = vectorize(dW2i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_network.layers[i].W2 += eps * dW2i
        f1 = residual_network.hidden_layers_forward(x)
        residual_network.layers[i].W2 -= eps * dW2i
        f2 = residual_network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_network.JacW2iMv(x, vec_dW2i, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual network with respect to W2 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_network_Jacb1Mv_test(residual_network, i):
    
    """
    Preforms the Jacobian test on a residual network with respect to b1 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    x = np.random.randn(residual_network.input_size, 1)
    db1i = np.random.randn(*residual_network.layers[i].b1.shape)
    db1i = db1i / np.linalg.norm(db1i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_network.layers[i].b1 += eps * db1i
        f1 = residual_network.hidden_layers_forward(x)
        residual_network.layers[i].b1 -= eps * db1i
        f2 = residual_network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_network.Jacb1iMv(x, db1i, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual network with respect to b1 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()

def residual_network_Jacb2Mv_test(residual_network, i):
    
    """
    Preforms the Jacobian test on a residual network with respect to b2 of the ith residual layer, and plots the results.

    Parameters:
    resnet: a residual network object

    Returns:
    None
    """

    x = np.random.randn(residual_network.input_size, 1)
    db2i = np.random.randn(*residual_network.layers[i].b2.shape)
    db2i = db2i / np.linalg.norm(db2i)
    eps = 1.

    E1 = []
    E2 = []

    for _ in range(20):
        residual_network.layers[i].b2 += eps * db2i
        f1 = residual_network.hidden_layers_forward(x)
        residual_network.layers[i].b2 -= eps * db2i
        f2 = residual_network.hidden_layers_forward(x)

        e1 = np.linalg.norm(f1 - f2)
        e2 = np.linalg.norm(f1 - (f2 + eps * residual_network.Jacb2iMv(x, db2i, i)))
        E1.append(e1)
        E2.append(e2)
        eps = eps * 0.5
    
    plt.title('Jacobian test for residual network with respect to b2 of the {}th residual layer'.format(i))
    plt.plot(E1, label='first order error')
    plt.plot(E2, label='second order error')
    plt.yscale('log')
    plt.legend()
    plt.show()


def hidden_layer_JacT_test(hidden_layer):
    """
    This test ensures that the hidden layer passes
    the Jacobian Transposed test, it assumes it already passed 
    the Jacobian tests for the methods JacxMv, JacWMv, JacbMv.
    and tests the JacxTMv, JacWTMv, JacbTMv methods.
    """

    x = np.random.randn(hidden_layer.input_size, 1)
    v1 = np.random.randn(hidden_layer.input_size, 1)
    u1 = np.random.randn(hidden_layer.output_size, 1)

    passx = np.dot(u1.T, hidden_layer.JacxMv(x, v1)) - np.dot(v1.T, hidden_layer.JacxTMv(x, u1)) < 1e-10

    if not passx:
        print('JacxTMv test failed')
        return False
    
    W = hidden_layer.W
    v2 = np.random.randn(W.shape[0] *W.shape[1], 1)
    u2 = np.random.randn(hidden_layer.output_size, 1)

    passW = np.dot(u2.T, vectorize(hidden_layer.JacWMv(x, v2))) - np.dot(v2.T, vectorize(hidden_layer.JacWTMv(x, u2))) < 1e-10

    if not passW:
        print('JacWTMv test failed')
        return False
    
    b = hidden_layer.b
    v3 = np.random.randn(b.shape[0], 1)
    u3 = np.random.randn(hidden_layer.output_size, 1)

    passb = np.dot(u3.T, hidden_layer.JacbMv(x, v3)) - np.dot(v3.T, hidden_layer.JacbTMv(x, u3)) < 1e-10

    if not passb:
        print('JacbTMv test failed')
        return False
    
    return True

def residual_layer_JacT_test(residual_layer):
    """
    This test ensures that the residual layer passes
    the Jacobian Transposed test, it assumes it already passed
    the Jacobian tests for the methods JacxMv, JacW1Mv, JacW2Mv, Jacb1Mv, Jacb2Mv.
    and tests the JacxTMv, JacW1TMv, JacW2TMv, Jacb1TMv, Jacb2TMv methods.
    """

    x = np.random.randn(residual_layer.input_size, 1)
    v1 = np.random.randn(residual_layer.input_size, 1)
    u1 = np.random.randn(residual_layer.input_size, 1)

    passx = np.dot(u1.T, residual_layer.JacxMv(x, v1)) - np.dot(v1.T, residual_layer.JacxTMv(x, u1)) < 1e-10

    if not passx:
        print('JacxTMv test failed')
        return False
    
    W1 = residual_layer.W1
    v2 = np.random.randn(W1.shape[0] * W1.shape[1], 1)
    u2 = np.random.randn(residual_layer.input_size, 1)

    passW1 = np.dot(u2.T, vectorize(residual_layer.JacW1Mv(x, v2))) - np.dot(v2.T, vectorize(residual_layer.JacW1TMv(x, u2))) < 1e-10

    if not passW1:
        print('JacW1TMv test failed')
        return False
    
    W2 = residual_layer.W2
    v3 = np.random.randn(W2.shape[0] * W2.shape[1], 1)
    u3 = np.random.randn(residual_layer.input_size, 1)

    passW2 = np.dot(u3.T, vectorize(residual_layer.JacW2Mv(x, v3))) - np.dot(v3.T, vectorize(residual_layer.JacW2TMv(x, u3))) < 1e-10

    if not passW2:
        print('JacW2TMv test failed')
        return False
    
    b1 = residual_layer.b1
    v4 = np.random.randn(b1.shape[0], 1)
    u4 = np.random.randn(residual_layer.input_size, 1)

    passb1 = np.dot(u4.T, residual_layer.Jacb1Mv(x, v4)) - np.dot(v4.T, residual_layer.Jacb1TMv(x, u4)) < 1e-10

    if not passb1:
        print('Jacb1TMv test failed')
        return False
    
    b2 = residual_layer.b2
    v5 = np.random.randn(*b2.shape)
    u5 = np.random.randn(residual_layer.input_size, 1)

    passb2 = np.dot(u5.T, residual_layer.Jacb2Mv(x, v5)) - np.dot(v5.T, residual_layer.Jacb2TMv(x, u5)) < 1e-10

    if not passb2:
        print('Jacb2TMv test failed')
        return False
    
    return True