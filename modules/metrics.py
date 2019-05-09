import numpy as np

def coefficient_of_determination(_y, _predicted):
    
    y = np.asarray(_y)
    predicted = np.asarray(_predicted)

    y_mean = y.mean()

    numerator_sum = 0
    denominator_sum = 0

    for id_val, val in enumerate(y):
        numerator_sum += (predicted[id_val] - y_mean) * (predicted[id_val] - y_mean)
        denominator_sum += (val - y_mean) * (val - y_mean)
    
    return numerator_sum / denominator_sum