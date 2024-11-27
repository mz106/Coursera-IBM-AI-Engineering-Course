# Used to 
# Takes the dependent variable (y-axis values), and result of function apply_linear_regression()

def find_residual_error(y_data, predicted):
    residuals = []

    for y in y_data:
        index_of_y = y_data.index(y)
        result = y - predicted[index_of_y]
        residuals.append(float(result))

    return residuals