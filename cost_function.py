
from theoretical_function import make_photocurrent_func


def make_cost_func(compiled_data):

    def cost_function(eta, x0):
        photo_current_func = make_photocurrent_func(eta, x0)
        cost = 0.0

        for x, y_tuple in compiled_data.iteritems():
            y_sum, y_sum_squared, y_count = y_tuple
            y_predicted = photo_current_func(x)
            y_mean = y_sum / y_count
            y_error = y_mean - y_predicted
            cost += y_error * y_error / y_sum_squared

        return cost

    return cost_function
