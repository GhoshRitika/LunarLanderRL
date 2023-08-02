import numpy as np
import matplotlib.pyplot as plt

def plot_learning_curve(x, scores, figure_file):
    plt.clf()
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)

def norm_action(env, ac):
        high = env.action_space.high
        low = env.action_space.low
        return 2.0 * (ac - low) / (high - low) - 1.0

def normalize_action(ac, low, high):
        return 2.0 * (ac - low) / (high - low) - 1.0

def add_actions(env, ac1, ac2):
    ac = ac1 + ac2
    return np.maximum(np.minimum(ac, env.action_space.high),
                        env.action_space.low)