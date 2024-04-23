import numpy as np
from scipy.optimize import minimize
from scipy.special import erf

# Define the objective function (loss)
def maxwell_loss(x, E0, B0):
    # Unpack the input array x into electric and magnetic field components
    Ex, Ey, Ez, Bx, By, Bz = x.reshape((6,))

    # Compute the Maxwell's equations loss function
    loss = 0.5 * (E0**2 + B0**2) - np.dot(x[:3], E0) - np.dot(x[3:], B0)
    return loss

# Define the gradient of the objective function
def maxwell_gradient(x, E0, B0):
    # Unpack the input array x into electric and magnetic field components
    Ex, Ey, Ez, Bx, By, Bz = x.reshape((6,))

    # Compute the gradient of the Maxwell's equations loss function
    grad = np.zeros_like(x)
    grad[:3] = -E0 + 2 * np.dot(Ex, E0) / (1 + erf(E0))
    grad[3:] = -B0 + 2 * np.dot(Bx, B0) / (1 + erf(B0))
    return grad

# Define the initial guess for the optimization
x0 = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# Define the parameters of Maxwell's equations
E0 = 10.0
B0 = 5.0

# Perform gradient descent optimization
res = minimize(maxwell_loss, x0, args=(E0, B0), method="CG", jac=maxwell_gradient)

# Print the optimized solution
print(res.x)
