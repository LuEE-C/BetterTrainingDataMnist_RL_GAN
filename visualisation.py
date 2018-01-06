import matplotlib.pyplot as plt
import pickle
import numpy as np

def exponential_average(old, new, b1):
    return old * b1 + (1-b1) * new

if __name__ == '__main__':
    values = pickle.load(open('Agent/values.pkl', 'rb'))

    new_values, old = [], 0
    for val in values:
        old = exponential_average(old, val, 0.9)
        new_values.append(old)

    plt.plot(new_values)
    plt.show()

    values = pickle.load(open('Agent/test_values.pkl', 'rb'))

    new_values, old = [], 0
    for val in values:
        old = exponential_average(old, val, 0.9)
        new_values.append(old)

    plt.plot(new_values)
    plt.show()

    values = pickle.load(open('Agent/critic_loss.pkl', 'rb'))

    new_values, old = [], 0
    for val in values:
        old = exponential_average(old, val, 0.9)
        new_values.append(old)

    plt.plot(new_values)
    plt.show()


    values = pickle.load(open('Agent/policy_loss.pkl', 'rb'))

    new_values, old = [], 0
    for val in values:
        old = exponential_average(old, val, 0.9)
        new_values.append(old)

    plt.plot(new_values)
    plt.show()
