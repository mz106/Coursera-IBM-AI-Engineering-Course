# takes the y-intercept (theta_0), the x coefficient (theta_1), and the independent variables (e.g. x-values)
# used to apply the linear regression model - either to test the training set or make predictions

def apply_regression_model(theta_0, theta_1, x_data):
    predicted = []

    for x in x_data:
        val = theta_0 + (theta_1 * x)
        predicted.append(float(val))
    
    return predicted