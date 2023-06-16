import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import math

# read .csv into DataFrame
house_data = pd.read_csv("house_prices.csv")
size = house_data['sqft_living']
price = house_data['price']

# ML handles arrays not Data Frames so convert it into arrays
x = np.array(size).reshape(-1, 1)
y = np.array(price).reshape(-1, 1)

#  we use Linear Regression + fit() in the training
model = LinearRegression()
model.fit(x, y)

# MSE and R value
regression_model_mse = mean_squared_error(x, y)
print("MSE: ", math.sqrt(regression_model_mse))
print("R squared value: ", model.score(x, y))

# we can get the b value after the model fit
# this is the b0
print(model.intercept_[0])
# this is the b1 in our model
print(model.coef_[0])

# visualize the data-set with the fitted model
plt.scatter(x, y, color='green')
plt.plot(x, model.predict(x), color='black')
plt.title('Linear Regression')
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()

print("Prediction by the model: ", model.predict([[2000]]))
