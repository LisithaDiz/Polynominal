import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(0)
x = 2 - 3 * np.random.normal(0, 1, 20)

y = x - 2 * (x ** 2) + 0.5 * (x ** 3) + np.random.normal(-3, 3, 20)

x = x[:, np.newaxis]
y = y[:, np.newaxis]

model = LinearRegression()
model.fit(x, y)
y_predict = model.predict(x)


def RMSE(w, b):
    error = 0
    for i in range(len(x)):
        error += (y[i] - (x[i] * w + b)) ** 2

    return (error / len(x)) ** 0.5


mean_of_features = np.mean(y)


def R_2_score(w, b):
    RSS = 0
    TSS = 0
    for i in range(len(x)):
        RSS += (y[i] - (x[i] * w + b)) ** 2  # sum of squares of residuals
        TSS += (y[i] - np.mean(y)) ** 2  # total sum of squares
    return 1 - RSS / TSS


weight = model.coef_[0]
intercept = model.intercept_
print(weight, intercept)
print(RMSE(weight, intercept))
print(R_2_score(weight, intercept))


print(np.sqrt(mean_squared_error(y, y_predict)))
print(r2_score(y, y_predict))

plt.scatter(x, y, s=10)
plt.scatter(0, intercept, color='y')

plt.plot(x, y_predict, color='red')
plt.show()
