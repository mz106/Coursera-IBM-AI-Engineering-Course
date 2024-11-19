import pandas as pd 
import matplotlib.pylab as plt 
import numpy as np 

path = "./auto.csv"

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df = pd.DataFrame(pd.read_csv(path, names = headers))

# replace "?" with NaN

df.replace("?", np.nan, inplace = True)

#Evaluate for missing data - gives a boolean for each cell in dataframe

missing_data = df.isnull()
# print(missing_data.head())

# count missing values in each column

# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())
#     print("")

# normalize data
# replace by mean - normalized-losses, stroke, bore, horsepower, peak-rpm

# calculate mean value for normalized-losses
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis = 0)
print("Avg of norm loss", avg_norm_loss)

# replace NaN with mean value in normalized-losses column
df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace = True)

# replace by frequency - num-of-doors

# drop whole row - price

