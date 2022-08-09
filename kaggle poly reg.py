import math
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

x = np.arange(0.0, 2000) - 1000

a = 0.00000000001
b = 0
c = 0
d = 0
e = -5
f = 0
y = a * np.power(x, 5) +( e *x)
plt.scatter(x,y)
plt.show()
