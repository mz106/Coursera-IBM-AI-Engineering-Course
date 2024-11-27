# Used to find the Mean Squared Error
# Takes reuslt of function find_residual_error() - a list
# returns a real


def find_mae(residual_data):
    total = 0

    for res in residual_data:
        
        
        result = abs(res)
        
        total += result

    
    return total / len(residual_data)