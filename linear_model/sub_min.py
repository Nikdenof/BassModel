import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

n1 = 65000 
n2 = n1 # Коэффициенты эта, характеризующие цену продукции

prediction_ru = np.genfromtxt("outputs/base_prediction_ru.csv", delimiter=",")
prediction_en = np.genfromtxt("outputs/base_prediction_en.csv", delimiter=",")
coefficients_ru = np.genfromtxt("outputs/coef_ru.csv", delimiter=",")
coefficients_en = np.genfromtxt("outputs/coef_en.csv", delimiter=",")
# Цель субсидии Q - увеличение продаж в 2030 году на 30 % в сравнении с прогнозом
q = 1.30 * prediction_ru[-1]
print("Цель субсидии Q =", q)

# Определение субсидии

# Есть цель Q, нужна ступенчатая функция для s.
sub_start_ru = prediction_ru[4]
sub_start_en = prediction_en[4]
t = 10


def calc_x(start_ru, start_en, s_t):
    a1, b1, y1 = coefficients_ru
    a2, b2, y2 = coefficients_en
    x1_i = a1 - b1 * (n1 - start_ru - s_t) + y1 * (n2 - start_en)
    x2_i = a2 - b2 * (n2 - start_en) + y2 * (n1 - start_ru - s_t)
    return x1_i, x2_i

def lin_model(s):
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

def objective(s):
    _, s_lst = lin_model(s)
    return sum(s_lst)

def sub_model(s):
    x1, _ = lin_model(s)
    return x1[-1]

# Ограничение - результат должен быть равен Q 
def constr1(s):
    return sub_model(s) - q

def constr2(s):
    return s[1] - s[0]*1.1

def constr3(s):
    return s[2] - s[1]*1.1


con1 = {'type': 'eq', 'fun': constr1}

con2 = {'type': 'ineq', 'fun': constr2}

con3 = {'type': 'ineq', 'fun': constr3}

cons = [con1, con2, con3]

# Процесс оптимизации
methods_yak = ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
b1 = (0, 500)
b2 = (0, 500)
b3 = (0, 500)
bnds = (b1, b2, b3)
x0 = np.zeros(3) # предположительные значения субсидии s1, s2, s3
x0[0], x0[1], x0[2] = 150, 200, 350
options = {"maxiter": 5000}
sol = minimize(objective, x0 = x0, method = methods[8], bounds = bnds, constraints = cons, options=options)
print(sol) # Печатаем результат с оценкой работы оптимизатора

# Сравнение изначального прогноза и результата субсидии
plt.plot(np.arange(5, 15), lin_model(sol.x)[0], 'r',  label = 'Выходные данные модели с учетом субсидии')
plt.plot(np.arange(len(prediction_ru)), prediction_ru, label = 'Выходные данные модели без учета субсидии')
plt.plot(len(prediction_ru) - 1, q, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green", label = "Цель субсидии Q") 
plt.title(label = 'Сравнение изначального прогноза и результата субсидии')
plt.legend()
plt.savefig('outputs/sub_res.png')
plt.show()

# Subsidy step change plot
plt.step(np.arange(10), lin_model(sol.x)[1])
plt.title(label = "Измение размера субсидии в период ее действия")
plt.savefig('outputs/sub_step.png')
plt.show()
