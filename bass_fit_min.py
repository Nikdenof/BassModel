import numpy as np
from model_bass import BassModel

sum_foreign, cumsum_foreign= np.genfromtxt('data/processed/sales_en.csv', delimiter=',') 
sum_russian, cumsum_russian = np.genfromtxt('data/processed/sales_ru.csv', delimiter=',') 

bass_russian = BassModel(cumsum_russian, sum_russian, cumsum_foreign, sum_foreign)

lst = bass_russian.fit(num_iterations=20000)
result = bass_russian.predict(num_years=5, visualize=False)
print(result)

base_prediction = bass_russian.predict(num_years=15)
print(f"Базовый прогноз через 10 лет = {base_prediction[-1]}")

# Цель субсидии Q - увеличение продаж в 2030 году на 30 % в сравнении с прогнозом
subsidy_goal = 1.3 * base_prediction[-1][0]
print("Цель субсидии Q =", subsidy_goal)

# Есть цель Q, нужна ступенчатая функция для s.
subsidy_start= base_prediction[4]
subsidy_length = 10 # in years
subsidy_steps = 5


bass_russian.set_subsidy(subsidy_goal, subsidy_length, subsidy_steps)


def constr2(s):
    return s[-1] - s[-2]*0.9

def constr3(s):
    return s[2] - s[1]*0.8

con2 = {'type': 'ineq', 'fun': constr2}

con3 = {'type': 'ineq', 'fun': constr3}
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
print(bass_russian.subsidy_minimize(method = methods[8], visualize=True, subsidy_upper_bound = 18000))


