import pandas as pd 
import matplotlib as plt 
from matplotlib import pyplot
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

# ======= Question 1 ===========
# Based on the example above, replace NaN in columns with mean value

# normalize data
# replace by mean - normalized-losses, stroke, bore, horsepower, peak-rpm

# calculate mean value for normalized-losses
avg_norm_loss = df["normalized-losses"].astype("float").mean(axis = 0)
# print("Avg of norm loss", avg_norm_loss)

# replace NaN with mean value in normalized-losses column
# df["normalized-losses"].replace(np.nan, avg_norm_loss, inplace = True)

df["normalized-losses"] = df["normalized-losses"].replace(np.nan, avg_norm_loss)

# calculate mean value for bore and replace NaN with mean
avg_bore = df["bore"].astype("float").mean(axis = 0)
# print("avg bore: ", avg_bore)

# df["bore"].replace(np.nan, avg_bore, inplace = True)
df["bore"] = df["bore"].replace(np.nan, avg_bore)

# calculate mean value for stroke and replace NaN with mean

avg_stroke = df["stroke"].astype("float").mean(axis = 0)
# print("avg stroke: ", avg_stroke)

df["stroke"] = df["stroke"].replace(np.nan, avg_stroke)

# calculate mean value for horsepower and replace NaN with mean

avg_horsepow = df["horsepower"].astype("float").mean(axis = 0)
# print("avg horsepow: ", avg_horsepow)

df["horsepower"] = df["horsepower"].replace(np.nan, avg_horsepow)

# calculate mean value for peak-rpm and replace NaN with mean
avg_peak_rpm = df["peak-rpm"].astype("float").mean(axis = 0)
# print("avg_peak_rpm: ", avg_peak_rpm)

df["peak-rpm"] = df["peak-rpm"].replace(np.nan, avg_peak_rpm)

# replace by frequency - num-of-doors

# print(df["num-of-doors"].value_counts())

# print(df["num-of-doors"].value_counts().idxmax())

df["num-of-doors"] = df["num-of-doors"].replace(np.nan, "four")

# drop whole row - price

df.dropna(subset = ["price"], axis = 0, inplace = True)

# reset index
df.reset_index(drop = True, inplace = True)

# print(df.head())

# Checks --- check datatypes --- some need changing

# print(df.dtypes)

df[["bore", "stroke"]] = df[["bore", "stroke"]].astype("float")
df[["normalized-losses"]] = df[["normalized-losses"]].astype("int")
df[["price"]] = df[["price"]].astype("float")
df[["peak-rpm"]] = df[["peak-rpm"]].astype("float")

# print(df.dtypes)

# ======= Question 2 ===========
# According to the example below, transform mpg to L/100km in the column of "highway-mpg" and 
# change the name of the column to "highway-L/100km"

# example convert mpg to L/100km in city-mpg column

# print(df.head())

city_mpg_converted = 235 / df["city-mpg"]

df["city-mpg"] = city_mpg_converted
df = df.rename(columns = { "city-mpg": "city-L/100km" })

# df["city-L/100km"] = 235 / df["city-mpg"]


# print(df.head())

# Question - convert highway-mpg

# print(df.head())

# =============== Question 3 =========================
# Normalization
# According tto the lenght/width example below, normalize the column height 
# This method is called "Max Normalization" or "scaling by the maximum"

highway_val_converted = 235 / df["highway-mpg"]

df["highway-mpg"] = highway_val_converted
df = df.rename(columns = { "highway-mpg": "highway-L/100km" })

# print(df.head())
# print(df["length"].head())
df["length"] = df["length"] / df["length"].max()
# print(df["length"].head())
df["width"] = df["width"] / df["width"].max()

# Max normalize the height column

# print(df["height"].head())
df["height"] = df["height"] / df["height"].max()
# print(df["height"].head())

# print(df[["length", "width", "height"]].head())

# =========== Binning ================

# placing horsepower values into three bins - Low, Medium and High
df["horsepower"] = df["horsepower"].astype(int, copy = True)

plt.pyplot.hist(df["horsepower"])

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
# plt.pyplot.savefig('images/horse_power_hist_1.png')
# plt.pyplot.show

bins = np.linspace(min(df["horsepower"]), max(df["horsepower"]), 4)

group_names = ['Low', 'Medium', 'High']

df['horsepower-binned'] = pd.cut(df['horsepower'], bins, labels = group_names, include_lowest = True)

# print(df['horsepower-binned'].value_counts())

plt.pyplot.hist(df["horsepower"], bins = 3)

plt.pyplot.xlabel("horsepower")
plt.pyplot.ylabel("count")
plt.pyplot.title("horsepower bins")
# plt.pyplot.savefig('images/horse_power_hist_2.png')
# plt.pyplot.show

# ============ Indicator variables ===============
# a dummy variable, a numerical value, which has no inherent meaning but are used to label categories
# for example, regression does not understand words, only numbers. So, if we have 2 categories 
# 'gas' and 'disel', we must assign numerical values to each e.g. gas = 1, disel = 0

# print(df.columns)

# get indicator variables and assign it to dataframe "dummy_variable_1" - use .astype() in a coding env
# print(df['fuel-type'].head())
dummy_variable_1 = pd.get_dummies(df['fuel-type']).astype('int')
# print(dummy_variable_1.head())

# change column names for clarity

dummy_variable_1.rename(columns = { 'gas': 'fuel-type-gas', 'diesel': 'fuel-type-diesel' }, inplace = True)

# print(dummy_variable_1)

# merge data frame with dummy variable
df = pd.concat([df, dummy_variable_1], axis = 1)

# drop original fuel-type column from df
df.drop("fuel-type", axis = 1, inplace = True)

# print(df.head())

# ============ Question 4 ============

# create indicator variable for column 'aspiration'

print(df['aspiration'].head())

dummy_variable_2 = pd.get_dummies(df['aspiration']).astype('int')
# print(dummy_variable_2.head())

dummy_variable_2.rename(columns = { 'std': 'standard-aspiration', 'turbo': 'turbo-aspiration' }, inplace = True)
# print(dummy_variable_2.head())

# =========== Question 5 ============

# merge to df and drop column aspiration

df = pd.concat([df, dummy_variable_2], axis = 1)
df.drop('aspiration', axis = 1, inplace = True)
print(df.head())

# ============ save to csv ==================

df.to_csv('./clean_auto.csv')