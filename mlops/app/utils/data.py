import numpy as np
import matplotlib.pyplot as plt

def generate_mean_and_std_dev_break_point():
    mean = np.random.uniform(-10, 10)  # Random mean between -10 and 10
    std_dev = np.random.uniform(0.1, 1)  # Random standard deviation between 0.1 and 5
    break_point = np.random.randint(50, 100)  # Random break point between 10 and 50 iterations
    return mean, std_dev, break_point

def plot_observations_with_breaks(list_observations, list_breaks):
    plt.figure(figsize=(10, 5))
    plt.plot(list_observations)
    plt.title('Observations Over Time')
    plt.xlabel('Iteration')
    plt.ylabel('Observation Value')
    for break_point in list_breaks:
        plt.axvline(x=break_point, color='r', linestyle='--', label='Break Points' if break_point == list_breaks[0] else "")
    plt.legend()
    plt.savefig('observations_plot.png')
    plt.show()
