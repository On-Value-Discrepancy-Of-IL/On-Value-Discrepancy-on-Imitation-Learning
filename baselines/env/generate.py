import numpy as np

x_range = (1.0, 1.5)
y_range = (0.45, 1.05)
z_range = (0.45, 0.65)

num_objects = 1000
xs = np.random.uniform(x_range[0], x_range[1], size=num_objects)
ys = np.random.uniform(y_range[0], y_range[1], size=num_objects)
zs = np.random.uniform(z_range[0], z_range[1], size=num_objects)

goals = []
for i in range(num_objects):
    goals.append([xs[i], ys[i], zs[i]])

goals = np.array(goals)
f = open('fetch_goal.npy', 'wb')
np.save(f, goals)
