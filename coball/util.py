import sys
import numpy as np

import matplotlib
# matplotlib.use("MacOSX")
# matplotlib.use("Agg")
from matplotlib import pyplot as plt


def print_progress(count, step, loss, scores, total=1000, bar_len = 60):
    if count == 0:
        return

    filled_len = int(round(bar_len * step / float(total)))

    percents = round(100.0 * step / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    window = scores[-100:] if len(scores) > 100 else scores
    mean = np.mean(window) if len(window) > 0 else 0.0
    min = np.min(window) if len(window) > 0 else 0.0
    max = np.max(window) if len(window) > 0 else 0.0

    sys.stdout.write("{}/{} [{}] loss: {:+.3E} / score: {:+.3f}({:+.0f}/{:+.0f}/{:+.0f})\r"\
            .format(count, step, bar, loss, mean, min, scores[-1], max))
    sys.stdout.flush()


def plot(data, windows=[1,10,100], path="", colors=['r', 'g', 'c'], labels=["1", "10", "100"]):
    """Plot the provided array with running averages with the given window sizes."""
    if path is not None and path != "":
        start_plot()

    for window, color, label in zip(windows, colors, labels):
        plt.plot(range(len(data)), [ _mean(data, i, window) for i in range(len(data)) ],
                c=color, label=label)

    if path is None:
        return
    elif path == "":
        plt.show()
    else:
        save_plot(path)

def start_plot():
    plt.figure()

def save_plot(path, loc='upper left'):
    plt.legend(loc=loc)
    plt.savefig(path)

def _mean(data, i, window):
    return np.mean(data[i-window:i]) if i > window else np.mean(data[:i])
