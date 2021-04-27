import matplotlib.pyplot as plt
import numpy as np

A = np.array([[1, 2, 3], [1, 4, 9], [7, 8, 9]])
print(A.shape)
print(A[:,:2])
fig, ax = plt.subplots()
heatmap = ax.pcolor(A, cmap='YlOrRd')
fig.colorbar(heatmap, ax=ax)
plt.show()

