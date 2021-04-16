import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
ax = fig.add_subplot()
ax.scatter(x, y)
ax.set_xlabel('X')
ax.set_ylabel('Y')
fig.savefig('example_scatterplot.pdf')