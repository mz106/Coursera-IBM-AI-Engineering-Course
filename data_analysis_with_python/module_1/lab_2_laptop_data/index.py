import pandas as pd
import numpy as np

# === Task 1: Import csv and create dataframe ===
path = "./laptop_pricing_dataset_base.csv"

df = pd.read_csv(path, header = None)

# print(df.head())

# === Task 2: Add headers === 

headers = ["Manufacturer", "Category", "Screen", "GPU", "OS", "CPU__core", "Screen_Size_inch", "CPU_frequency", "RAM_GB", "Storage_GB_SSD", "Weight_kg", "Price"]

df.columns = headers
df.columns
# print(df.head(10))

# === Task 3: Replace "?" with NaN ===

df.replace("?", np.NaN, inplace = True)


# print(df1.head())

# === Task 4: prinf dataframe datatypes

# print(df.dtypes)

# === Task 5: Print statistical description of dataset using .describe()

# without NaN
# print(df.describe())

# with NaN
# print(df.describe(include = "all"))

# include objet datatype
# print(df.describe(include = "object"))

# === Task 6: Print summary of info in dataset using .info()

print(df.info())