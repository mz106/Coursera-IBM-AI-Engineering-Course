import json

def predict_score(data, hours, theta_0, theta_1, mae):
    
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