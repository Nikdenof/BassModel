from scipy.optimize import curve_fit 

class BassModel:
    def __init__(self, base_cumsum, base_sum, competetor_cumsum, competetor_sum):
        self.base_cumsum = base_cumsum
        self.base_sum = base_sum
        self.competetor_cumsum = competetor_cumsum
        self.competetor_sum = competetor_sum

    @staticmethod
    def bass_function(x, p, q, m):
        """
        This is a base bass function that has 3 parameters:
            m - the number of people estimated to eventually adopt the new product
            q - the coefficient of imitation
            p - the coefficient of innovation
        """
        return p * m + (q - p) * x - (q / m) * x ** 2

    @staticmethod
    def bass_function_modified(x, p, q, m):
        """
        This is a modified bass function that has 3 parameters:
            m - the number of people estimated to eventually adopt the new product
            q - the coefficient of imitation
            p - the coefficient of innovation
        """
        return p * m + (q - p) * x - (q / m) * x ** 2

    def fit(self, visualize=False, num_iterations=5000):
        """
        This functions performes curve_fit function from the scipy module
        in order to find the coefficients of the bass function.
        Returns an array of size equal to the number of coefficients
        """
        fit_coefficients, _ = curve_fit(self.bass_function, self.base_cumsum[0:5], self.base_sum[1:6], maxfev=num_iterations)
        self.coeff_m, self.coeff_q, self.coeff_p = fit_coefficients
        if visualize:
            print('test')
        return fit_coefficients

    def minimize(self, visualize=False):
        pass
