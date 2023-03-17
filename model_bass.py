import numpy as np
from scipy.optimize import curve_fit 
# from plots import fit_plt


class BassModel:
    def __init__(self, base_cumsum, base_sum, competetor_cumsum, competetor_sum) -> None:
        self.base_cumsum = base_cumsum
        self.base_sum = base_sum
        self.competetor_cumsum = competetor_cumsum
        self.competetor_sum = competetor_sum

    @staticmethod
    def join_two_lists(lst1: list, lst2: list) -> np.ndarray:
        """
        This functions joins two list in order to pass it to bass function
        """
        return np.append(lst1, lst2) 

    @staticmethod
    def split_joined_list(lst_join: list, lst_len: int) -> tuple[list, list]:
        """
        This function splits the output of the bass model into two equal sized join_two_lists
        """
        lst1 = lst_join[:lst_len]
        lst2 = lst_join[lst_len:]
        return lst1, lst2

    def bass_function(self, x: float, p1: float, q1: float, m1: float, 
                      p2: float, q2: float, m2: float, q12: float) -> list[float, float]:
        """
        This is a base bass function that has 3 parameters:
            m1, m2 - the number of people estimated to eventually adopt the new product
            q1, q2  - the coefficient of imitation
            p1, p2 - the coefficient of innovation
            q12 - competative coefficient
        It returns the result of calculated bass equation
        """
        array_length = len(x) // 2
        x1, x2 = self.split_joined_list(x, array_length) # dim(10,1) -> dim(5,2) 
        bass1 = (p1 + (q1 / m1) * (x1)) * (m1 - x1) + (q12 / m1) * x1 * (m2 - x2) 
        bass2 = (p2 + (q2 / m2) * (x2)) * (m2 - x2) + (q12 / m2) * x2 * (m1 - x1)

        bass_joined = self.join_two_lists(bass1, bass2) #dim(5,2) -> dim(10,1) 
        
        return bass_joined 

    @staticmethod
    def sum_calculate(lst1: list, lst2: list) -> list:
        sum_lst = []
        for x1, x2 in lst1, lst2:
            sum_run = x1 + x2
            sum_lst.append(sum_run)
        return sum_lst 

    def fit(self, num_iterations: int=5000) -> list[float]:
        """
        This functions performes curve_fit function from the scipy module
        in order to find the coefficients of the bass function.
        Returns an array of size equal to the number of coefficients
        """
        array_length = len(self.base_cumsum)
        cumsum_joined = self.join_two_lists(self.base_cumsum[:array_length-1], self.competetor_cumsum[:array_length-1])
        sum_joined = self.join_two_lists(self.base_sum[1:array_length], self.competetor_sum[1:array_length])
        self.fit_coefficients, _ = curve_fit(self.bass_function, cumsum_joined, sum_joined, maxfev=num_iterations)
        return self.fit_coefficients

    def predict(self, num_years, visualize=False) -> np.ndarray:
        """ 
        Calculates prediction based on computed coefficients, returns an array, containing cummulitive sum of bass function
        """
        # Need to provide info in the manner: [competetor_cumsum, base_cumsum]1 .. [c_cms, bs_cms]2 .. 
        cumsum_joined = self.join_two_lists(self.base_cumsum[:array_length-1], self.competetor_cumsum[:array_length-1])
        for t in range(num_years):
            running_sum = self.bass_function(cumsum_joined[t], *self.fit_coefficients)
            running_cumsum = cum_sum[t] + running_sum
            cum_sum.append(running_cumsum)
        self.fit_predict = cum_sum
        return cum_sum
#
#    def calc_err(self):
#        pass
#    def minimize(self, visualize=False):
#        pass
