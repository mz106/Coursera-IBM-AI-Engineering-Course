import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

path = "./clean_auto.csv"
path2 = "./auto.csv"

df = pd.read_csv(path)


headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

df2 = pd.DataFrame(pd.read_csv(path2, names = headers))

# print(df2.head())

df2.replace("?", np.nan, inplace = True)

# ================ Correlation =========================

# ================== Question 1 ==================

# What is the data type of the column "peak-rpm"?

# print(df["peak-rpm"].dtypes)

# Answer - float64

# =================== Question 2 ================

# Find the correlation between the following columns: bore, stroke, compression-ratio, horsepower

# print(df.dtypes)

# print(df[["price", "fuel-type-diesel", "fuel-type-gas"]].corr())

# print(df[["bore", "stroke", "compression-ratio", "horsepower"]].corr())

# print(df[["price", "horsepower"]].corr())

# =================== Continuous Numerical Variables =====================

# Continous numerical variables are variable that may contain any value within some range.
# Can be int64 or float64

# We can use 'regplot' to plot a scatter plot and fit a regression line. 

# Example: Positive linear relationship

# sns.regplot( x = "engine-size", y = "price", data = df)
# plt.ylim(0,)
# plt.savefig('images/scatter_1.png')

# print(df[["engine-size", "price"]].corr())

# sns.regplot( x = "highway-L/100km", y = "price", data = df)
# plt.ylim(0, )
# plt.savefig('images/scatter_2.png')

# print(df[["highway-L/100km", "price"]].corr())
# print(df2[["highway-mpg", "price"]].corr())

# sns.regplot( x = "highway-mpg", y = "price", data = df2)
# plt.ylim(0,)
# plt.savefig("images/scatter_3.png")

# ========= Example: Weak Linear Relationship ===========

# Is peak-rpm a predictor of price?

# print(df[["peak-rpm", "price"]].corr())

#           peak-rpm     price
# peak-rpm  1.000000 -0.101616
# price    -0.101616  1.000000

# The correlation between peak-rpm and price is -0.101616. 

# sns.regplot( x = "peak-rpm", y = "price", data = df)
# plt.ylim(0,)
# plt.savefig("images/scatter_3.png")

# As seen in images/scatter_3.png, the regression line is alomst horizontal, and the plots
# are far from the fitted line. So, with the correlation, peak-rpm is not a good
# predictor of price. 

# ================= Question 3 ===============

# Find the correlation between x = "stroke" and y = "price"

# print(df[["stroke", "price"]].corr())

# correlation: 0.082269 - that's low!

# sns.regplot( x = "stroke", y = "price", data = df)
# plt.ylim(0,)
# plt.savefig("images/scatter_4.png")

# Whilst there is a slight positive correlation, we see on scatter_4.png that
# many of the plots do no converge on or near the regression line.
# Therefore, stroke is not a good indicator of price. 

# ========= Categorical values ===========

# This is comparing values that are not numerical, in the sense of a value. 
# Instead it is somethng like "detached house" vs "price" or "semi-detached" vs "price".

# sns.boxplot(x = "body-style", y = "price", data = df)
# plt.savefig("images/boxplot_1.png")

# sns.boxplot(x = "engine-location", y = "price", data = df)
# plt.savefig("images/boxplot_2.png")

# sns.boxplot(x = "drive-wheels", y = "price", data = df)
# plt.savefig("images/boxplot_3.png")

# =============== Descriptive statistical analysis ==============

# print(df.describe())

# print(df.describe(include="object"))

# value count
# print(df["drive-wheels"].value_counts())

df_drive_wheel_count = df["drive-wheels"].value_counts().to_frame()
df_drive_wheel_count = df_drive_wheel_count.reset_index()
df_drive_wheel_count = df_drive_wheel_count.rename(columns={"drive-wheels": "value_counts"})
df_drive_wheel_count.index.name = "drive-wheels"
# print(df_drive_wheel_count)

engine_loc_counts = df["engine-location"].value_counts().to_frame()
engine_loc_counts = engine_loc_counts.reset_index()
engine_loc_counts = engine_loc_counts.rename(columns={"engine-location": "value-counts"})
engine_loc_counts.index.name = "engine-location"
print(engine_loc_counts)