import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(initial_theta,eta,epsilon=1e-8):
    theta = initial_theta
    theta_history.append(initial_theta)

    while True:
        gradiant = dJ(theta)
        last_theta = theta
        theta -= eta * gradiant
        theta_history.append(theta)
        if (abs(J(theta) - J(last_theta)) < epsilon):
            break

def plot_theta_history():
    plt.plot(plot_x,J(plot_x))
    plt.plot(np.array(theta_history), J(np.array(theta_history)), color='r', marker="+")
    plt.show()
    print(theta)
    print(J(theta))
    print('cal times=', len(theta_history))