import numpy as np
from scipy.optimize import curve_fit 
from plots import fit_plt

class LinearModel:
    def __init__(self, base_cumsum, competetor_cumsum, coeff_eta1=65000, coeff_eta2=65000):
        self.base_cumsum = base_cumsum
        self.competetor_cumsum = competetor_cumsum
        self.coeff_eta1 = coeff_eta1
        self.coeff_eta2 = coeff_eta2

    @staticmethod
    def x_t(X, a, b, y, n1, n2):
        """
        This is a base linear model that has 3 parameters:
        """
        x1, x2 = X # Кумулятивные продажи первого и второго продукта
        return a - b * (n1 - x1) + y * (n2 - x2) 

    def fit(self, num_iterations=5000):
        """
        This functions performes curve_fit function from the scipy module
        in order to find the coefficients of the linear model.
        Returns an array of size equal to the number of coefficients
        """
        #####
        ##### CHANGE TO 4 points
        fit_coefficients, _ = curve_fit(self.bass_function, self.base_cumsum[0:5], self.base_sum[1:6], maxfev=num_iterations)
        self.coeff_p, self.coeff_q, self.coeff_m = fit_coefficients
        return fit_coefficients

#    def calc_prediction(self, num_years, start_value=0, visualize=False):
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
    def calc_x(start_ru, start_en, s_t):
        a1, b1, y1 = coefficients_ru
        a2, b2, y2 = coefficients_en
        x1_i = a1 - b1 * (n1 - start_ru - s_t) + y1 * (n2 - start_en)
        x2_i = a2 - b2 * (n2 - start_en) + y2 * (n1 - start_ru - s_t)
        return x1_i, x2_i
    def calculate_model(self):
        start_ru = sub_start_ru
        start_en = sub_start_en
        a1, b1, y1 = coefficients_ru
        a2, b2, y2 = coefficients_en
        x1 = []
        x2 = []
        s_lst = []
        s_t = s[0]
        for i in range(t):
            if i == 4:
                s_t = s[1]
            elif i == 7:
                s_t = s[2]
            x1_i, x2_i = calc_x(start_ru, start_en, s_t)
            x1.append(x1_i)
            x2.append(x2_i)
            s_lst.append(s_t)
            start_ru = x1_i
            start_en = x2_i
        return x1, s_lst

""" 
I need to also incorporate the minimization into the class
Maybe it is a good idea, maybe not
"""
