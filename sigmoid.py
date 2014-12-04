import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-5,5,10000)
print x

y = 1/(1 + np.exp(-x))
z = y*(1-y)
print y

plt.plot(x,y)
plt.figure()
plt.plot(x,z)
plt.show()

	