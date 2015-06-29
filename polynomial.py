import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def model(p):

    b_0 = 0
    b_1 = 57
    b_2 = 4600

    return b_0 + b_1 * p + b_2 * (p ** 2)

x = np.linspace(0, 50, 300)

model_vec = np.vectorize(model)

y = model_vec(x)

plt.plot(x, y)
plt.show()
