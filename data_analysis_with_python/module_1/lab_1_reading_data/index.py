import pandas as pd
import numpy as np

path = "./auto.csv"

df = pd.DataFrame(pd.read_csv(path))

# === Print first 5 rows ===
# print(df.head(5))

# === Print bottom 5 rows ===
# print(df.tail())

# === Create data headers list (column names) and add to the dataframe ===

headers = ["symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors", "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width", "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size", "fuel-system", "bore", "stroke", "compression-ratio", "horsepower", "peak-rpm", "city-mpg", "highway-mpg", "price"]

# print("headers\n", headers)

df.columns = headers
# print(df.columns)

# print(df.head())

# === Replace '?' with NaN === 

df1 = df.replace('?', np.NaN)

# print(df1.head(20))

# === Drop rows where price shows NaN === 

df = df1.dropna(subset=["price"], axis = 0)
# print(df.head(20))


# === Find the name of the columns === 

# print(df.columns)

# === Save data to new csv locally ===

# df.to_csv("auto_no_headers.csv", index = False)
# df.to_csv("auto_with_headers.csv", index = True)

# === Show df datatypes === 

# print(df.dtypes)

# === Get statistical summary of of each column using .describe() (excluding NaN) === 

# print(df.describe())

# === Get statistical summary of of each column using .describe() (indluding NaN) === 

# print(df.describe(include = "all"))

# === Select particular column(s) ("length", "compression-ratio") === 

print(df[["length", "compression-ratio"]] )

# === Print concise summary of the dataframe (shows index, column name, num of not null, datatype)

print(df.info())