import numpy as np
import matplotlib.pyplot as plt

def col_mean(M):
    """
    Computes the column-wise mean of a given matrix.

    This function calculates the mean of each row in the input matrix M, effectively reducing the dimensionality of M from (n, m) to (n, 1). Each element in the output matrix represents the average of a row from the input matrix.

    Parameters:
    M (numpy.ndarray): A matrix of size (n, m) where 'n' is the number of rows and 'm' is the number of columns.

    Returns:
    numpy.ndarray: A column vector of size (n, 1), where each element is the mean of the corresponding row in the input matrix M.

    Example:
    >>> M = np.array([[1, 2, 3],
                      [4, 5, 6]])
    >>> col_mean(M)
    array([[2],
           [5]])
    """
    return np.mean(M, axis=1, keepdims=True)


def SGD(Xt, Yt, Xv, Yv, NN, lr, batch_size=64, epochs=200):
    """
    Preforms mini-batch SGD on the given neural network.
    
    Parameters:
    Xt (numpy.ndarray): Training data matrix of size (n, m) where 'n' is the number of features and 'm' is the number of training samples.
    Yt (numpy.ndarray): Training labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of training samples.
    Xv (numpy.ndarray): Validation data matrix of size (n, m) where 'n' is the number of features and 'm' is the number of validation samples.
    Yv (numpy.ndarray): Validation labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of validation samples.
    NN (NeuralNetwork): The neural network to train.
    lr (float): The learning rate.
    batch_size (int): The batch size.
    epochs (int): The number of epochs.

    Returns:
    training_loss (list): A list of the training loss at each epoch.
    validation_loss (list): A list of the validation loss at each epoch.
    training_accuracy (list): A list of the training accuracy at each epoch.
    validation_accuracy (list): A list of the validation accuracy at each epoch.
    """

    # Initialize lists to store the loss and accuracy at each epoch
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []

    # fix batch size if it is not < number of samples
    if batch_size >= Xt.shape[1]:
        batch_size = Xt.shape[1]


    for epoch in range(epochs):
        # Shuffle the data
        indices = np.arange(Xt.shape[1])
        np.random.shuffle(indices)
        Xt = Xt[:, indices]
        Yt = Yt[:, indices]

        # Mini-batch SGD
        for i in range(0, Xt.shape[1], batch_size):
            Xb = Xt[:, i:i+batch_size]
            Yb = Yt[:, i:i+batch_size]
            # Backpropagation and update parameters
            gradients = NN.backpropagation(Xb, Yb)       
            NN.update_parameters(gradients, lr)

        # Compute the loss and accuracy on the training set
        loss = NN.loss(Xt, Yt)
        accuracy = NN.accuracy(Xt, Yt)
        print(f'Epoch {epoch}, training loss: {loss}')# TODO: comment this out
        print(f'Epoch {epoch}, training accuracy: {accuracy}')# TODO: comment this out
        training_loss.append(loss)
        training_accuracy.append(accuracy)

        # Compute the loss and accuracy on the validation set
        loss = NN.loss(Xv, Yv)
        accuracy = NN.accuracy(Xv, Yv)
        print(f'Epoch {epoch}, validation loss: {loss}')# TODO: comment this out
        print(f'Epoch {epoch}, validation accuracy: {accuracy}')# TODO: comment this out
        validation_loss.append(loss)
        validation_accuracy.append(accuracy)

    return training_loss, validation_loss, training_accuracy, validation_accuracy

def plot_data(NN, Xt, Yt, Xv, Yv):
    """
    Plots the data and the predictions of the given neural network.
    This function assumes that our data is 2-dimensional and our neural network is a classifier.

    Parameters:
    NN (NeuralNetwork): The neural network to plot.
    Xt (numpy.ndarray): Training data matrix of size (2, m) where 2 is the number of features and 'm' is the number of training samples.
    Yt (numpy.ndarray): Training labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of training samples.
    Xv (numpy.ndarray): Validation data matrix of size (2, m) where 2 is the number of features and 'm' is the number of validation samples.
    Yv (numpy.ndarray): Validation labels matrix of size (l, m) where 'l' is the number of classes and 'm' is the number of validation samples.

    Returns:
    None
    """

    # Compute the predictions of the neural network on the training and validation sets
    prediction_training = NN.forward(Xt)
    prediction_validation = NN.forward(Xv)

    plt.figure(figsize=(10, 8))
     
    # plot the points of Xt colored by their true class
    plt.subplot(2, 2, 1)
    plt.title('Training True')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(Yt, axis=0))

    # plot the points of Xt colored by their predicted class
    plt.subplot(2, 2, 2)
    plt.title('Training Prediction')
    plt.scatter(Xt[0], Xt[1], c=np.argmax(prediction_training, axis=1))

    # plot the points of Xv colored by their true class
    plt.subplot(2, 2, 3)
    plt.title('Validation True')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(Yv, axis=0))

    # plot the points of Xv colored by their predicted class
    plt.subplot(2, 2, 4)
    plt.title('Validation Prediction')
    plt.scatter(Xv[0], Xv[1], c=np.argmax(prediction_validation, axis=1))

    plt.show()

def plot_loss_and_accuracy(tloss, taccuracy, vloss, vaccuracy):
    """
    Plots the loss and accuracy of the training and validation sets.

    Parameters:
    tloss (list): A list of the training loss at each epoch.
    taccuracy (list): A list of the training accuracy at each epoch.
    vloss (list): A list of the validation loss at each epoch.
    vaccuracy (list): A list of the validation accuracy at each epoch.

    Returns:
    None
    """

    plt.figure(figsize=(10, 8))
    
    # Plot the loss on the training set
    plt.subplot(2, 2, 1)
    plt.title('Training Loss')
    plt.plot(tloss)

    # Plot the accuracy on the training set
    plt.subplot(2, 2, 2)
    plt.title('Training Accuracy')
    plt.plot(taccuracy)

    # Plot the loss on the validation set
    plt.subplot(2, 2, 3)
    plt.title('Validation Loss')
    plt.plot(vloss)

    # Plot the accuracy on the validation set
    plt.subplot(2, 2, 4)
    plt.title('Validation Accuracy')
    plt.plot(vaccuracy)

    plt.show()

def log(x):
    """
    A function that adds a small number to x to avoid taking the log of 0.

    Parameters:
    x (numpy.ndarray): The input data.

    Returns:
    numpy.ndarray: The log of x.
    """
    
    return np.log(x + 1e-10)

def vectorize(A):
    return np.reshape(A, (-1, 1))




