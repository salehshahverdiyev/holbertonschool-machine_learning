#!/usr/bin/env python3
'''
    Script Documentation
'''
import numpy as np
import matplotlib.pyplot as plt


def all_in_one():
    '''
        Function Documentation
    '''
    y0 = np.arange(0, 11) ** 3

    mean = [69, 0]
    cov = [[15, 8], [8, 15]]
    np.random.seed(5)
    x1, y1 = np.random.multivariate_normal(mean, cov, 2000).T
    y1 += 180

    x2 = np.arange(0, 28651, 5730)
    r2 = np.log(0.5)
    t2 = 5730
    y2 = np.exp((r2 / t2) * x2)

    x3 = np.arange(0, 21000, 1000)
    r3 = np.log(0.5)
    t31 = 5730
    t32 = 1600
    y31 = np.exp((r3 / t31) * x3)
    y32 = np.exp((r3 / t32) * x3)

    np.random.seed(5)
    student_grades = np.random.normal(68, 15, 50)

    plt.figure()  # figure size
    plt.suptitle("All in One", fontsize='x-small')  # Main title

    # Plot 1 (Line Graph)
    plt.subplot(3, 2, 1)  # Top left
    plt.plot(y0, color='red', linestyle='solid')
    plt.xlim(0, 10)
    plt.xlabel("x", fontsize='x-small')
    plt.ylabel("y", fontsize='x-small')
    plt.title("Task 0", fontsize='x-small')

    # Plot 2 (Scatter Plot)
    plt.subplot(3, 2, 2)  # Top right
    plt.scatter(x1, y1, color='magenta', marker='o')
    plt.xlabel("Height (in)", fontsize='x-small')
    plt.ylabel("Weight (lbs)", fontsize='x-small')
    plt.title("Task 1", fontsize='x-small')

    # Plot 3 (Logarithmic Line Graph)
    plt.subplot(3, 2, 3)  # Middle left
    plt.plot(x2, y2)
    plt.xlabel("Time (years)", fontsize='x-small')
    plt.ylabel("Fraction Remaining", fontsize='x-small')
    plt.title("Task 2", fontsize='x-small')
    plt.yscale("log")
    plt.xlim(0, 28650)

    # Plot 4 (Two Line Graphs)
    plt.subplot(3, 2, 4)  # Middle right
    plt.plot(x3, y31, color='red', linestyle='dashed', label='C-14')
    plt.plot(x3, y32, color='green', linestyle='solid', label='Ra-226')
    plt.xlabel("Time (years)", fontsize='x-small')
    plt.ylabel("Fraction Remaining", fontsize='x-small')
    plt.title("Task 3", fontsize='x-small')
    plt.xlim(0, 20000)
    plt.ylim(0, 1)
    plt.legend(loc='upper right', fontsize='x-small')

    # Plot 5 (Histogram)
    plt.subplot(3, 1, 3)  # Bottom row, spanning 2 columns
    plt.hist(student_grades, bins=np.arange(0, 101, 10), edgecolor='black')
    plt.xlabel("Grades", fontsize='x-small')
    plt.ylabel("Number of Students", fontsize='x-small')
    plt.title("Task 4", fontsize='x-small')

    plt.tight_layout()  # Adjust spacing
    plt.show()
