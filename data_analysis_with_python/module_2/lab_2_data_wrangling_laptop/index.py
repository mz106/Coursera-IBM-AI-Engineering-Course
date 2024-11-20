import numpy as np
import pandas as pd
import matplotlib as plt
from matplotlib import pyplot

path = './laptop_pricing_dataset_mod1.csv'

df = pd.read_csv(path, header = 0)

# print(df.info())
# print(df.columns)

# print(df.head())

# # # round column 'Screem_Size_cm' to 2dp

df[['Screen_Size_cm']] = np.round(df[['Screen_Size_cm']], 2)

# print(df.head())

# ============= Task 1 =============

# Evaluate the dataset for missing data

# missing_data = df.isnull()

# for column in missing_data.columns.values.tolist():
#     print(column)
#     print(missing_data[column].value_counts())
#     print("")

# ================= Task 2 ===============

# Replace missing values with mean
# replace "Weight_kg" missing values with mean of values

avg_weight_kg = df["Weight_kg"].astype("float").mean(axis = 0)
# print(avg_weight_kg)

df["Weight_kg"] = df['Weight_kg'].replace(np.nan, avg_weight_kg)
# print(df["Weight_kg"])

# Replace with most frequent value in "Screen_Size_cm"

# print("Screen Size calue count", df["Screen_Size_cm"].value_counts().idxmax())

val = df["Screen_Size_cm"].value_counts().idxmax()
# print("val", val)

df["Screen_Size_cm"] = df["Screen_Size_cm"].replace(np.nan, val)

# ================== Task 3 ============================

# Fixing data types of Weight_kg and Screen_Size_cm from object to float

df[["Weight_kg", "Screen_Size_cm"]] = df[["Weight_kg", "Screen_Size_cm"]].astype("float")

# ==================== Task 4 ========================

# data standardization - convert screen size to inches and weight to pounds
# print(df.head())
screen_size_to_inches = df["Screen_Size_cm"] / 2.54
df["Screen_Size_cm"] = screen_size_to_inches

weight_to_pounds = 2.205 * df["Weight_kg"]
df["Weight_kg"] = weight_to_pounds

df = df.rename(columns = { "Screen_Size_cm": "Screen_Size_in", "Weight_kg": "Weight_lbs" })

# print(df.head())

# Data normalization - use max normalization on column CPU_frequency

# print(df.head())
df["CPU_frequency"] = df["CPU_frequency"] / df["CPU_frequency"].max()
# print(df.head())

# ================= Task 5 ===================

# Binning - create three bins for "Price" - Low, Medium and High 
# New Attribute called "Price-binned"

price_bins = np.linspace(min(df["Price"]), max(df["Price"]), 4)
# print(price_bins)

group_names = ["Low", "Medium", "High"]

df["Price-binned"] = pd.cut(df["Price"], bins = price_bins, labels = group_names, include_lowest = True)

# print(df.head())

# plot bar graph for price bins

# bin_counts = df["Price-binned"].value_counts().sort_index()

# bin_counts.plot(kind = "bar", figsize = (8, 5))

pyplot.bar(group_names, df["Price-binned"].value_counts())

# pyplot.bar(df["Price-binned"], bins = 3, height = 5)
pyplot.xlabel("Price")
pyplot.ylabel("Count")
pyplot.title("Price Bins")

# pyplot.savefig('images/price_bins.png')

# =============== Task 6 ==============
# Indicator variables - convert Screen attribute to 2 indicator variables

# 1. "ScreeN-IPS_panel"
# 2. "Screen-Full_HD"
# 3. Drop Screen

dummy_var_1 = pd.get_dummies(df["Screen"]).astype("int")

# print(dummy_var_1)

dummy_var_1 = dummy_var_1.rename(columns = { "Full HD": "Screen-Full_HD", "IPS Panel": "Screen-IPS_panel" })

# print(dummy_var_1)

print(df.head())

df = pd.concat([df, dummy_var_1], axis = 1)

df = df.drop("Screen", axis = 1)

print(df.head())