from __future__ import division
import matplotlib.pyplot as plt
import math
import numpy as np


def normal_dist(x, mu, sigma):

    first_part = 1 / (sigma * math.sqrt(2 * math.pi))
    second_part = math.exp( - ( x - mu ) ** 2  / (2 * (sigma  ** 2)))
    
    return first_part * second_part



x = np.linspace(-5, 5, 1000)

normal_dist_vec = np.vectorize(normal_dist)

y = normal_dist_vec(x, 0, 1)


from sklearn.linear_model import LinearRegression


clf = LinearRegression()

X = [[0, 2, 4],
     [0, 2, 4]]
y = [5, 5]

clf.fit(X, y)
