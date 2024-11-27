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