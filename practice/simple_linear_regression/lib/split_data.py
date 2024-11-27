
import json
import random
import os

# Load the JSON data (list of dictionaries)
script_dir = os.path.dirname(__file__)
json_path = os.path.join(script_dir, '..', 'data.json')

with open(json_path, 'r') as file:
    data = json.load(file)

# Shuffle the data
random.seed(42)  # For reproducibility
random.shuffle(data)

# Split the data (80% training, 20% testing)
split_index = int(0.8 * len(data))
train_data = data[:split_index]
test_data = data[split_index:]

# Separate into X (input) and y (output) for both sets
X_train = [item['num_beds'] for item in train_data]
y_train = [item['price'] for item in train_data]

X_test = [item['num_beds'] for item in test_data]
y_test = [item['price'] for item in test_data]

with open('new_data.json', 'w') as file:
    json.dump(train_data, file, indent=4)

# Print the results
print("Training set size:", len(X_train))
print("Testing set size:", len(X_test))
print("First few training examples:", list(zip(X_train, y_train))[:5])
print("First few testing examples:", list(zip(X_test, y_test))[:5])