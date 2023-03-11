
import matplotlib.pyplot as plt
import numpy as np

def plot_lines(x, y, legend_items, ylim_min = 0.5, ylim_max= 1.05, title='line graph'):

    for index, y_line in enumerate(y):
        plt.plot(x, y_line, '--', label="curve {}".format(index))

    plt.legend(legend_items, loc = "upper left")
    plt.ylim(ylim_min, ylim_max)
    plt.title(title)
    plt.show()


















