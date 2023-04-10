import numpy as np
from scipy.optimize import curve_fit, minimize
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
        b = (0.001, 100000)
        self.fit_coefficients, _ = curve_fit(self.bass_function, cumsum_joined, sum_joined, maxfev=num_iterations, bounds=b)
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
        #sum_predicted = []
        # Need to provide info in the manner: [competetor_cumsum, base_cumsum]1 .. [c_cms, bs_cms]2 .. 
        # cumsum_joined = self.join_two_lists(self.base_cumsum[:array_length-1], self.competetor_cumsum[:array_length-1])
        for t in range(num_years):
            running_sum = self.bass_predict(cumsum_predicted[t], *self.fit_coefficients, self.m1, self.m2)
        #    sum_predicted.append(running_sum)
            running_cumsum = [x + y for x, y in zip(cumsum_predicted[t], running_sum)] 
            cumsum_predicted.append(running_cumsum)
        self.fit_predict = cumsum_predicted 
        if visualize:
            domestic, foreign = self.process_result(cumsum_predicted)
            self.fit_plt(domestic, self.base_cumsum, title = "Модель Басса - местный производитель")
            self.fit_plt(foreign, self.competetor_cumsum, title = "Модель Басса - иностранный производитель")
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
        plt.title(label = title)
        plt.legend()
#        plt.savefig(save)
        plt.show()


    def bass_subsidy(self, x: list, subsidy_t: float) -> list[float, float]:
        """
        This is a subsidy bass function that has these parameters:
            m1, m2 - the number of people estimated to eventually adopt the new product
            q1, q2  - the coefficient of imitation
            p1, p2 - the coefficient of innovation
            q12 - competative coefficient
            subsidy_t - subsidy value
        It returns the result of calculated bass equation
        """
        x1, x2 = x
        p1, q1, p2, q2, q12 = self.fit_coefficients
        m1, m2 = self.m1, self.m2
        bass1 = (p1 + (q1 / m1) * (x1)) * (m1 - x1 + subsidy_t) + (q12 / m1) * x1 * (m2 - x2) 
        bass2 = (p2 + (q2 / m2) * (x2)) * (m2 - x2) + (q12 / m2) * x2 * (m1 - x1 + subsidy_t)
        bass_joined = [bass1, bass2]

        return bass_joined


    def set_subsidy(self, goal_q: int, years: int, num_steps: int) -> None:
        """
        In this funtion we set the goal Q, number of years, during which the subsidy
        will be in effect. The behaviour of the subsidy is a step function.
        We can set number of steps using `num_steps` variable

        Example:
            10 years, `num_steps` == 3
            self.steps = [0, 3, 6]
        """
        if years < 5:
            raise Exception("Sorry, but the minimum subsidy length is 5 years")
        if num_steps >= years - 1 or num_steps < 1:
            raise ValueError("The number of steps for subsidy function should be less then number of years and greater than 0")
        steps_history = [0]
        calculated_step = years // num_steps
        for i in range(num_steps - 1):
            running_step = steps_history[i] + calculated_step
            steps_history.append(running_step)
        self.subsidy_steps = steps_history
        self.subsidy_years = years
        self.goal_q = goal_q
        print("This is our steps", steps_history)


    def subsidy_model(self, s: list) -> tuple[list, list]:
        """
        This function takes subsidy list.
        Outputs the subsidy values, and the calculated bass_function
        for each year of the subsidy
        """
        subsidy_begin = len(self.base_cumsum) - 1
#        bass_domestic_i = self.base_cumsum[subsidy_begin]
#        bass_foreign_i = self.competetor_cumsum[subsidy_begin]
        bass_domestic = [self.base_cumsum[subsidy_begin]]
        bass_foreign = [self.competetor_cumsum[subsidy_begin]]
        subsidy_list = []
        pointer_subsidy = 0
#            running_sum = self.bass_predict(cumsum_predicted[t], *self.fit_coefficients, self.m1, self.m2)
#            running_cumsum = [x + y for x, y in zip(cumsum_predicted[t], running_sum)] 
#            cumsum_predicted.append(running_cumsum)
        for i in range(self.subsidy_years):
            if i == self.subsidy_steps[pointer_subsidy]:
                subsidy_t = s[pointer_subsidy]
                pointer_subsidy += 1 if pointer_subsidy < len(self.subsidy_steps) - 1  else 0
            bass_domestic_i, bass_foreign_i = self.bass_subsidy([bass_domestic[i], bass_foreign[i]], subsidy_t)
            bass_domestic.append(bass_domestic[i] + bass_domestic_i)
            bass_foreign.append(bass_foreign[i] + bass_foreign_i)
            subsidy_list.append(subsidy_t)
        return bass_domestic[1:], subsidy_list


    def objective_function(self, s):
        """
        The returned value is later minimized, the lower - the better,
        while keeping all the constraints in place
        """
        _, subsidy_list = self.subsidy_model(s)
        return sum(subsidy_list)


    def last_year_output(self, s):
        """
        Last year of the subsidy, used as one of constraints,
        as we need it to be equal to Q
        """
        bass_domestic, _ = self.subsidy_model(s)
        return bass_domestic[-1]


    def constr1(self, s):
        """ 
        This constraint makes sure, that our final value is q
        """
        return self.last_year_output(s) - self.goal_q


    def subsidy_minimize(self, add_constraints: list = [], method: str = "trust-constr", num_iterations: int = 5000, 
                         subsidy_upper_bound: int = 3000, visualize: bool = False):
        x0 = np.zeros(len(self.subsidy_steps)) # предположительные значения субсидии s1, s2, s3

        base_bound = (0, subsidy_upper_bound)
        bounds = []
        for i in range(len(self.subsidy_steps)):
            bounds.append(base_bound)
        bounds = tuple(bounds)

        constraints = []
        q_constraint = {'type': 'eq', 'fun': self.constr1}
        constraints.append(q_constraint)
        if add_constraints:
            constraints.extend(add_constraints)

        options = {"maxiter": num_iterations}

        self.solution = minimize(self.objective_function, x0 = x0, method = method, bounds = bounds, constraints = constraints, options=options)
        if visualize:
            self.solution_plot(self.solution)
        return self.solution

    
    def solution_plot(self, solution) -> None:
        end_point = len(self.base_cumsum) + self.subsidy_years
        plt.plot(np.arange(end_point - len(self.base_cumsum)), self.subsidy_model(solution.x)[0], 'r-*',  label = 'Выходные данные модели с учетом субсидии')
#        plt.plot(np.arange(len(self.base_cumsum), end_point), self.subsidy_model(solution.x)[0], 'r',  label = 'Выходные данные модели с учетом субсидии')
#        plt.plot(np.arange(len(self.base_cumsum)), self.base_cumsum, label = 'Выходные данные модели без учета субсидии')
#        plt.plot(end_point - 1, self.goal_q, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green", label = "Цель субсидии Q") 
        plt.plot(end_point - 1 - len(self.base_cumsum), self.goal_q, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green", label = "Цель субсидии Q") 
        plt.title(label = 'Сравнение изначального прогноза и результата субсидии')
        plt.legend()
        plt.show()

        plt.step(np.arange(self.subsidy_years), self.subsidy_model(solution.x)[1])
        plt.title(label = "Измение размера субсидии в период ее действия")
        plt.show()

    def calc_err(self):
        pass
