import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

from sklearn import linear_model
from sklearn.metrics import r2_score

path = "./FuelConsumptionCo2.csv"

df = pd.DataFrame(pd.read_csv(path))

# print(df.head())

cdf = df[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', "CO2EMISSIONS"]]

# print(cdf.head())

viz = cdf[['CYLINDERS', 'ENGINESIZE', "CO2EMISSIONS", 'FUELCONSUMPTION_COMB']]

# print(viz.hist())

# plt.savefig('images/figure_1.png')
# plt.show()

# Scatter plot for fuel consumption/co2 emissions
# plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.savefig('images/fuel_to_emissions_scatter.png')
# plt.show()

# Scatter plot for engine size/co2emissions
# plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
# plt.xlabel("FUELCONSUMPTION_COMB")
# plt.ylabel("Emission")
# plt.savefig('images/engine_to_emissions_scatter.png')
# plt.show()

# Scatter plot for cylinders/emissions
# plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
# plt.xlabel("Cylinders")
# plt.ylabel("Emission")
# plt.savefig('images/cylinders_to_emissions_scatter.png')
# plt.show()

# Create training and test data
# Create mask

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr = linear_model.LinearRegression()
train_x = np.asanyarray(train[["ENGINESIZE"]])
train_y = np.asanyarray(train[["CO2EMISSIONS"]])
regr.fit(train_x, train_y)

print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_) 

# plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color="blue")
# plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], "-r")
# plt.xlabel("Engine size")
# plt.ylabel("Emission")
# plt.savefig('images/train_engine_emissions_fit.png')

# Evaluate using engine size

# test_x = np.asanyarray(test[["ENGINESIZE"]])
test_y = np.asanyarray(test[["CO2EMISSIONS"]])
# test_y_ = regr.predict(test_x)

# print("Mean absloute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
# print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
# print("R2-score: %.2f" % r2_score(test_y , test_y))

# Evaluate using fuel consumption

train_x = np.asanyarray(train[["FUELCONSUMPTION_COMB"]])
test_x = np.asanyarray(test[["FUELCONSUMPTION_COMB"]])

regr = linear_model.LinearRegression()
regr.fit(train_x, train_y)

print("Coefficients: ", regr.coef_)
print("Intercept: ", regr.intercept_)

predictions = regr.predict(test_x)

mae = np.mean(np.absolute(predictions - test_y))
print("MAE: ", mae)