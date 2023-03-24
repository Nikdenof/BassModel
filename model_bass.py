import numpy as np
from scipy.optimize import curve_fit 
import matplotlib.pyplot as plt


class BassModel:
    def __init__(self, base_cumsum, base_sum, competetor_cumsum, competetor_sum) -> None:
        self.base_cumsum = base_cumsum
        self.base_sum = base_sum
        self.competetor_cumsum = competetor_cumsum
        self.competetor_sum = competetor_sum
        self.m1 = 85000
        self.m2 = 100000

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

    def bass_function(self, x: list, p1: float, q1: float,
                      p2: float, q2: float, q12: float) -> list[float, float]:
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
        m1 = self.m1
        m2 = self.m2

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

    def fit(self, num_iterations: int=5000, train_test_split: float = 0.85) -> list[float]:
        """
        This functions performes curve_fit function from the scipy module
        in order to find the coefficients of the bass function.
        Returns an array of size equal to the number of coefficients
        """
        
        train_length = int(len(self.base_cumsum) * train_test_split)
        cumsum_joined = self.join_two_lists(self.base_cumsum[:train_length-1], self.competetor_cumsum[:train_length-1])
        sum_joined = self.join_two_lists(self.base_sum[1:train_length], self.competetor_sum[1:train_length])
        print(train_length)
        print(len(self.base_cumsum))
        self.fit_coefficients, _ = curve_fit(self.bass_function, cumsum_joined, sum_joined, maxfev=num_iterations)
        self.train_length = train_length
        return self.fit_coefficients

    @staticmethod
    def bass_predict(x: list, p1: float, q1: float, 
                      p2: float, q2: float, q12: float, m1: float, m2: float) -> list[float, float]:
        """
        This is a base bass function that has 3 parameters:
            m1, m2 - the number of people estimated to eventually adopt the new product
            q1, q2  - the coefficient of imitation
            p1, p2 - the coefficient of innovation
            q12 - competative coefficient
        It returns the result of calculated bass equation
        """
        x1, x2 = x
        bass1 = (p1 + (q1 / m1) * (x1)) * (m1 - x1) + (q12 / m1) * x1 * (m2 - x2) 
        bass2 = (p2 + (q2 / m2) * (x2)) * (m2 - x2) + (q12 / m2) * x2 * (m1 - x1)

        bass_joined = [bass1, bass2]

        return bass_joined


    def predict(self, num_years, visualize=False) -> list:
        """ 
        Calculates prediction based on computed coefficients, returns an array, containing cummulitive sum of bass function
        """
        cumsum_predicted = [[self.base_cumsum[0], self.competetor_cumsum[0]]] 
#        cumsum_predicted = [[0, 0]] 
        sum_predicted = []
        # Need to provide info in the manner: [competetor_cumsum, base_cumsum]1 .. [c_cms, bs_cms]2 .. 
        # cumsum_joined = self.join_two_lists(self.base_cumsum[:array_length-1], self.competetor_cumsum[:array_length-1])
        for t in range(num_years):
            running_sum = self.bass_predict(cumsum_predicted[t], *self.fit_coefficients, self.m1, self.m2)
            sum_predicted.append(running_sum)
            running_cumsum = [x + y for x, y in zip(cumsum_predicted[t], running_sum)] 
            cumsum_predicted.append(running_cumsum)
        self.fit_predict = cumsum_predicted 
        if visualize:
            domestic, foreign = self.process_result(cumsum_predicted)
            self.fit_plt(domestic, self.base_cumsum, title = "Placeholder")
            self.fit_plt(foreign, self.competetor_cumsum, title = "Placeholder2")
        return self.fit_predict 


    @staticmethod
    def process_result(result: list) -> tuple[np.ndarray, np.ndarray]:
        data = np.array(result)
        data = data.transpose()
        data_1, data_2 = data
        return data_1, data_2


    def fit_plt(self, data, train_data, title) -> None:
        plt.plot(np.arange(self.train_length-1), data[:self.train_length - 1],'g--',  label = 'Аппроксимация модели') # Correct version -1 -> none
        plt.plot(np.arange(self.train_length-2, len(data)), data[self.train_length-2:], 'r', label = 'Предсказание модели') #Correct version -2 -> -1
        plt.scatter(np.arange(len(train_data)), train_data, label = 'Исходные данные')
#        plt.plot(np.arange(3, len(cs)), model_output[2: len(cs)-1], 'r',  label = 'Предсказания модели')
#        plt.plot(np.arange(1, 4), model_output[:3], 'g--',  label = 'Аппроксимация модели')
#        plt.plot(np.arange(2), [cs[0], model_output[0]], 'g--')
#        plt.scatter(np.arange(6), cs, label = 'Исходные данные')
        plt.title(label = title)
        plt.legend()
#        plt.savefig(save)
        plt.show()


    def calc_err(self):
        pass

    def minimize(self, visualize=False):
        pass
