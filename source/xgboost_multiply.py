import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb


size = 10000
X = np.zeros((size, 2))

# Z = np.meshgrid(np.linspace(-1, 1, 100), np.linspace(-1, 1, 100))
Z = np.meshgrid(np.linspace(-1.05, 0.95, 100), np.linspace(-1.05, 0.95, 100))

X[:, 0] = Z[0].flatten()
X[:, 1] = Z[1].flatten()

y_mul = X[:, 0] * X[:, 1]
y_div = X[:, 0] / X[:, 1]

name, op = ('MULTIPLICATION', y_mul)
fig = plt.figure(figsize=(15, 10))
ax = fig.gca(projection='3d')
ax.set_title(name)
ax.plot_trisurf(X[:, 0], X[:, 1], op, cmap=plt.cm.viridis, linewidth=0.2)
# plt.show()
# plt.savefig("{}.jpg".format(name))


mdl = xgb.XGBRegressor()
mdl.fit(X, op)

fig = plt.figure(figsize=(15, 10))
ax = fig.gca(projection='3d')
ax.set_title("{} - NOISE = 0".format(name))
ax.plot_trisurf(X[:, 0], X[:, 1], mdl.predict(X), cmap=plt.cm.viridis, linewidth=0.2)
# plt.show()
# plt.savefig("{}_noise0.jpg".format(name))
