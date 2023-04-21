import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)
y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)


def loss_func(w, b):
    tot_error = 0
    for i in range(len(x)):
        tot_error += (y[i] - w * x[i] + b) ** 2
    return tot_error / float(len(x))


def grad_decent(w_now, b_now, L):
    w_grad_decent = 0
    b_grad_decent = 0
    n = len(x)

    for i in range(n):
        w_grad_decent += (-2 / n) * x[i] * (y[i] - (w_now * x + b_now))
        b_grad_decent += (-2 / n) * (y[i] - (w_now * x[i] + b_now))

        w = w_now - w_grad_decent * L
        b = b_now - b_grad_decent * L
        return w, b


w = 0
b = 0
L = 0.001
metrix_x = np.zeros(200)
metrix_y = np.zeros(200)
epochs = 200

for i in range(epochs):
    w, b = grad_decent(w, b, L)
    metrix_x[i] = i
    # metrix_y[i] = loss_func(w, b)

# print(w, b)
print(loss_func(w, b))
plt.scatter(x, y, s=10)
plt.show()
