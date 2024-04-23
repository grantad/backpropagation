import numpy as np
keras.models import Sequential
from keras.layers import Dense
from scipy.optimize import minimize

# Define the objective function (loss)
def loss(nn_params):
    # Unpack the neural network parameters
    W1, b1, W2, b2 = nn_params

    # Create a new neural network with these parameters
    model = Sequential()
    model.add(Dense(4, weights=[W1, np.zeros(W1.shape[1])], bias_init=lambda: b1))
    model.add(Dense(1, weights=[W2, np.zeros(W2.shape[1])], bias_init=lambda: b2))

    # Define the input data
    x = np.array([[0.05, 0.05], [0.10, 0.10], [0.15, 0.15], [0.20, 0.20]])
    y = np.array([0.01, 0.04, 0.07, 0.10])

    # Calculate the output of the neural network
    output = model.predict(x)

    # Calculate the mean squared error loss
    loss_value = (1/4) * np.sum((output - y)**2)

    return loss_value

# Define the gradient function
def grad(nn_params):
    # Unpack the neural network parameters
    W1, b1, W2, b2 = nn_params

    # Create a new neural network with these parameters
    model = Sequential()
    model.add(Dense(4, weights=[W1, np.zeros(W1.shape[1])], bias_init=lambda: b1))
    model.add(Dense(1, weights=[W2, np.zeros(W2.shape[1])], bias_init=lambda: b2))

    # Define the input data
    x = np.array([[0.05, 0.05], [0.10, 0.10], [0.15, 0.15], [0.20, 0.20]])
    y = np.array([0.01, 0.04, 0.07, 0.10])

    # Calculate the output of the neural network
    output = model.predict(x)

    # Calculate the error in each output
    errors = output - y

    # Calculate the gradients for each parameter
    dW1 = np.zeros_like(W1)
    db1 = np.zeros_like(b1)
    dW2 = np.zeros_like(W2)
    db2 = np.zeros_like(b2)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            for k in range(4):
                for l in range(1):
                    dW1[i, j] += errors[i] * x[i, j] * 1
                    db1[j] += errors[i] * 1
                    dW2[k, l] += errors[i] * 1 * 1
                    db2[l] += errors[i] * 1

    return [dW1.flatten(), db1.flatten(), dW2.flatten(), db2.flatten()]

# Initialize the neural network parameters
init_params = np.array([np.random.rand(4, 2), np.random.rand(4), np.random.rand(4, 1), np.random.rand(1)])

# Run gradient descent
res = minimize(loss, init_params, method="SLSQP", jac=grad)

# Print the final parameters
A great start!
