#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np
import matplotlib.pyplot as plt


def bars():
    '''
        Function Documentation
    '''
    np.random.seed(5)
    fruit = np.random.randint(0, 20, (4, 3))
    plt.figure(figsize=(6.4, 4.8))

    # labels
    labels = ['apples', 'bananas', 'oranges', 'peaches']
    people = ['Farrah', 'Fred', 'Felicia']

    # colors
    colors = ['red', 'yellow', '#ff8000', '#ffe5b4']

    # stacked bar graph
    plt.bar(people, fruit[0], label=labels[0], color=colors[0], width=0.5)
    plt.bar(people, fruit[1],
            bottom=fruit[0], label=labels[1], color=colors[1], width=0.5)
    plt.bar(people, fruit[2],
            bottom=fruit[0] + fruit[1], label=labels[2], color=colors[2],
            width=0.5)
    plt.bar(people, fruit[3],
            bottom=fruit[0] + fruit[1] + fruit[2], label=labels[3],
            color=colors[3], width=0.5)

    # labels and title
    plt.ylabel("Quantity of Fruit")
    plt.title("Number of Fruit per Person")

    # y-axis limits and ticks
    plt.ylim(0, 80)
    plt.yticks(np.arange(0, 81, 10))

    # legend
    plt.legend()
    plt.show()
