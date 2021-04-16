import numpy as np
import matplotlib.pyplot as plt

s = 0.90    #Success rate
num = 100    # # random tests


#Lists that store results
res1 = np.zeros(num)
res2 = np.zeros(num)
temp = np.zeros(10)
#temp_c = np.zeros(12)
#temp_c[11] = 1.0

for k in range(num):
    for r in range(10):
        rand = np.random.rand()
        if rand < 0.95:
            a = 1.0
        else:
            a = 0.0
        temp[r] = a
#        temp_c[r] = a
    res1[k] = temp.sum() / 10
#    res2[k] = temp_c.sum() / 12

rate1 = res1.mean()
#rate2 = res2.mean()
#err1 = (rate1 - s) / s
#err2 = (rate2 - s) / s
"""
y, x = np.histogram(rate1, range=(0, 1))
print(x)
print(y)
"""

d = np.random.rand(50)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(d, bins=10)
ax.set_title('first histogram')
ax.set_xlabel('x')
ax.set_ylabel('freq')
fig.show()
