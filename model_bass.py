import numpy as np
from scipy.optimize import curve_fit 
from plots import fit_plt

class BassModel:
    def __init__(self, base_cumsum, base_sum, competetor_cumsum, competetor_sum) -> None:
        self.base_cumsum = base_cumsum
        self.base_sum = base_sum
        self.competetor_cumsum = competetor_cumsum
        self.competetor_sum = competetor_sum

    @staticmethod
    def bass_function(X: list, p1: float, q1: float, m1: float, 
                      p2: float, q2: float, m2: float, q12: float) -> tuple[float, float]:
        """
        This is a base bass function that has 3 parameters:
            m - the number of people estimated to eventually adopt the new product
            q - the coefficient of imitation
            p - the coefficient of innovation
        It returns the result of calculated bass equation
        """
        x1, x2 = X
        bass1 = (p1 + (q1 / m1) * (x1)) * (m1 - x1) + (q12 / m1) * x1 * (m2 - x2) 
        bass2 = (p2 + (q2 / m2) * (x2)) * (m2 - x2) + (q12 / m2) * x2 * (m1 - x1)
        
        return bass1, bass2
#        return (p1 + (q1 / m1) * (x1)) * (m1 - x1), (p1 + (q1 / m1) * (x2)) * (m1 - x1)
    # BASS 1[0] + BASS 2[0] == SUM of their real value
    # NEED TO Come up with second value that will be BASS1/BASS2 and will be 
    # SUM!!!!!!!!!!!!!!!!!!!! OF TWO VALUES
    # NOT CUMSUM 

    def fit(self, num_iterations: int=5000) -> list[float]:
        """
        This functions performes curve_fit function from the scipy module
        in order to find the coefficients of the bass function.
        Returns an array of size equal to the number of coefficients
        """
        fit_coefficients, _ = curve_fit(self.bass_function, self.base_cumsum[0:5], self.base_sum[1:6], maxfev=num_iterations)
        self.coeff_p, self.coeff_q, self.coeff_m = fit_coefficients
        return fit_coefficients

#    def predict(self, arg) -> return_type:
#        """ Calculates prediction based on computed coefficients, returns an array, containing cummulitive sum of bass function"""
#        cum_sum = [start_value]
#        for t in range(num_years):
#            running_sum = self.bass_function(cum_sum[t], self.coeff_p, self.coeff_q, self.coeff_m)
#            running_cumsum = cum_sum[t] + running_sum
#            cum_sum.append(running_cumsum)
#        self.fit_predict = cum_sum
#        if visualize:
#            fit_plt(self.base_cumsum, self.fit_predict, title="Sasik", save="outputs/test.png")
#        return cum_sum
#
#    def calc_err(self):
#        pass
#    def minimize(self, visualize=False):
#        pass
