import pandas as pd

import json

from training_data import return_training_data
from test_data import return_test_data

# data = {
#     "hours_studied": [1, 2, 3, 4, 5],
#     "test_score": [52, 55, 60, 62, 68]
# }

#prediction 1 - 1hr:  {'raw_prediction': 51.6, 'with_positive_mae': 52.32, 'with_negative__mae': 50.88, 'variance_range': 1.4399999999999977} prediction 2 - 9hrs:  {'raw_prediction': 82.80000000000001, 'with_positive_mae': 83.52000000000001, 'with_negative__mae': 82.08000000000001, 'variance_range': 1.4399999999999977}

# data = return_training_data()
# result with training data
# prediction 1 - 1hr:  {'raw_prediction': 60.12888573717347, 'with_positive_mae': 61.611239885313374, 'with_negative__mae': 58.646531589033565, 'variance_range': 2.9647082962798095} prediction 2 - 9hrs:  {'raw_prediction': 137.8095512684796, 'with_positive_mae': 139.2919054166195, 'with_negative__mae': 136.3271971203397, 'variance_range': 2.9647082962798095}

# data = return_test_data()
data = return_training_data()

# prediction 1 - 1hr:  {'raw_prediction': 60.3403162544109, 'with_positive_mae': 61.69526256754522, 'with_negative__mae': 58.98536994127658, 'variance_range': 2.7098926262686405} prediction 2 - 9hrs:  {'raw_prediction': 139.12129835676734, 'with_positive_mae': 140.47624466990166, 'with_negative__mae': 137.76635204363302, 'variance_range': 2.7098926262686405}

# Step 1: Add data to DataFrame using Pandas
df = pd.DataFrame(data)

#    hours_studied (X)  test_score (Y)
# 0              1          52
# 1              2          55
# 2              3          60
# 3              4          62
# 4              5          68

# Step 2: Find mean of X and Y

mean_hrs_studied = df["hours_studied"].mean()
mean_test_scores = df["test_score"].mean()

# Step 3: Calculate Slope (theta_1)



def find_theta_1(x_data, x_bar, y_data, y_bar):
    numerator = 0
    denominator = 0

    for x in x_data:
        x_minus_avg = x - x_bar

        index_of_x = x_data.index(x)
        
        y_minus_avg = y_data[index_of_x] - y_bar
        
        numerator += x_minus_avg * y_minus_avg

        denominator += x_minus_avg ** 2
    
    return numerator / denominator

# Step 4: Calculate y-intercept (theta_0)

def find_theta_0(y_bar, theta_1, x_bar):
    
    result_1 = theta_1 * x_bar
    result_2 = y_bar - result_1

    return result_2



theta_1 = find_theta_1(data["hours_studied"], mean_hrs_studied, data["test_score"], mean_test_scores)

theta_0 = find_theta_0(mean_test_scores, theta_1, mean_hrs_studied)

# Step 5: Test against current dataset

def validate_test_score(theta_0, theta_1, x_data):
    predicted_scores = []

    for x in x_data:
        val = theta_0 + (theta_1 * x)
        predicted_scores.append(float(val))
    
    return predicted_scores

validation_results = validate_test_score(theta_0, theta_1, data["hours_studied"])

# Step 6: Find the residual error for each result

def find_residual_error(y_data, validated):
    residuals = []

    for y in y_data:
        index_of_y = y_data.index(y)
        result = y - validated[index_of_y]
        residuals.append(float(result))

    return residuals

residual_errors = find_residual_error(data["test_score"], validation_results)

# Step 7: Find Mean Absloute Error (MAE)

def find_mae(residual_data):
    total = 0

    for res in residual_data:
        
        
        result = abs(res)
        
        total += result

    
    return total / len(residual_data)

mae = find_mae(residual_errors)

# Step 8: Predict test score and adjust with MAE

def predict_score(data, hours, theta_0, theta_1, mae):
    print("mae!!!!!!: ", mae)
    score = theta_0 + (theta_1 * hours)

    positive_mae = score + mae
    negative_mae = score - mae
    variance_range = abs(positive_mae - negative_mae)

    

    results = {
        "raw_prediction": float(score),
        "with_positive_mae": float(positive_mae),
        "with_negative__mae": float(negative_mae),
        "variance_range": float(variance_range)
    }

    if data["is_training"]:
        with open("training_result.json", "w") as file:
            json.dump(results, file)
    elif data["is_test"]:
        with open("test_result.json", "w") as file:
            json.dump(results, file)

    return results

prediction_one = predict_score(data, 1, theta_0, theta_1, mae)
prediction_two = predict_score(data, 9, theta_0, theta_1, mae)

print("prediction 1 - 1hr: ", prediction_one, "prediction 2 - 9hrs: ", prediction_two)

