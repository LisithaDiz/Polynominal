import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error

for i in range(10):
    np.random.seed(0)
    X = 4 * np.random.rand(100, 1) - 2
    y = 4 + 2 * X + 5 * X ** 2 + 6 * X ** 3 + 2 * X ** 4 + 20 * np.random.rand(100, 1)

    poly_features = PolynomialFeatures(degree=i+1, include_bias=False)
    X_poly = poly_features.fit_transform(X)

    reg = LinearRegression()
    reg.fit(X_poly, y)

    X_vals = np.linspace(-2, 2, 100).reshape(-1, 1)
    X_vals_poly = poly_features.transform(X_vals)

    y_vals = reg.predict(X_vals_poly)

    print('r2 = ' , r2_score(y, y_vals))
    print(np.sqrt(mean_squared_error(y, y_vals)))

plt.scatter(X, y)
plt.plot(X_vals, y_vals, color='red')
plt.show()
