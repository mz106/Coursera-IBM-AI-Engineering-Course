import random
import json

# Function to generate a realistic dataset
def generate_realistic_housing_data(num_samples=100):
    data = []
    for _ in range(num_samples):
        num_beds = random.randint(1, 5)  # Bedrooms range from 1 to 5
        # Price range varies by the number of bedrooms
        if num_beds == 1:
            price = random.randint(70000, 150000)
        elif num_beds == 2:
            price = random.randint(150000, 250000)
        elif num_beds == 3:
            price = random.randint(250000, 400000)
        elif num_beds == 4:
            price = random.randint(400000, 600000)
        else:  # num_beds == 5
            price = random.randint(600000, 800000)
        data.append({"price": price, "num_beds": num_beds})
    return data

# Function to add a specified proportion of outliers
def add_proportional_outliers(data, outlier_percentage=0.2):
    total_outliers = int(len(data) * outlier_percentage)  # Calculate number of outliers
    outliers_per_bedroom = total_outliers // 5  # Divide among 1 to 5 bedrooms
    
    for num_beds in range(1, 6):  # Bedrooms 1 to 5
        for _ in range(outliers_per_bedroom):
            # Add expensive outlier
            data.append({"price": random.randint(800000, 1000000), "num_beds": num_beds})
            # Add cheap outlier
            data.append({"price": random.randint(50000, 100000), "num_beds": num_beds})
    return data

# Generate the initial dataset
housing_data = generate_realistic_housing_data()

# Add outliers to make them 20% of the total data
housing_data_with_proportional_outliers = add_proportional_outliers(housing_data, outlier_percentage=0.2)

# Save the data to a JSON file
with open("data.json", "w") as file:
    json.dump(housing_data_with_proportional_outliers, file, indent=4)

print("Data has been written to data.json")

lm = LinearRegression()
lm.fit(X, Y)
lm.score(X, Y)