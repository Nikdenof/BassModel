import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize


# Загрузка начальных данных
table_name = "SAPR.xlsx"
table = pd.read_excel(table_name, skiprows = 2, header = None)
table = table.transpose()[1:]
table.columns = ["Foreign", "Domestic", "Sum"]
table["Years"] = table.index

cad_en = table[["Years", "Foreign"]].copy()
cad_ru = table[["Years", "Domestic"]].copy()
cad_sum = table[["Years", "Sum"]].copy()


# Выделяем столбец кумулятивной суммы
def cum_sum_add(df):
    df = df.rename(columns={df.columns[1]: "revenues"})
    df['cum_sum'] = df['revenues'].cumsum()
    arr = df['revenues'].cumsum().values
    return df, arr


cad_en, cs_en = cum_sum_add(cad_en)
cad_ru, cs_ru = cum_sum_add(cad_ru)
cad_sum, cs_sum = cum_sum_add(cad_sum)

# Задаем коэффициенты эта, которые характеризуют цену продукции
n1 = 65000
n2 = n1


# Получение коэффициентов линейной модели спроса
def x_t(X, a, b, y):
    x1, x2 = X # Кумулятивные продажи первого и второго продукта
    return a - b * (n1 - x1) + y * (n2 - x2) 


# Ограничиваем диапазон для коэффициентов (>0) 
b = (0.001, 100000)
bounds = (b, b, b)
popt_en, _ = curve_fit(x_t, (cs_en[0:3], cs_ru[0:3]), cs_en[1:4], bounds = b)
popt_ru, _ = curve_fit(x_t, (cs_ru[0:3], cs_en[0:3]), cs_ru[1:4], bounds = b)


# Определяем аппроксимирующую модель
def fit_model(cs_1, cs_2, popt_1, popt_2, t = 5):
    cs1_fit = []
    cs2_fit = []
    a1, b1, y1 = popt_1
    a2, b2, y2 = popt_2
    cs_1, cs_2 = cs_1[0], cs_2[0]

    for i in range(t):
        c1 = x_t((cs_1, cs_2), a1, b1, y1)
        c2 = x_t((cs_2, cs_1), a2, b2, y2)
        cs1_fit.append(c1)
        cs2_fit.append(c2)
        cs_1, cs_2 = c1, c2

    return cs1_fit, cs2_fit


def plt_fit(cs_1, cs_2, popt_1, popt_2, title, save):
    plt.plot(np.arange(4, 6), fit_model(cs_1, cs_2, popt_1, popt_2)[0][3:], 'r',  label = 'Предсказания модели')
    plt.plot(np.arange(1, 5), fit_model(cs_1, cs_2, popt_1, popt_2)[0][:4], 'g--',  label = 'Аппроксимация модели')
    plt.plot(np.arange(2), [cs_1[0], fit_model(cs_1, cs_2, popt_1, popt_2)[0][0]], 'g--')
    plt.scatter(np.arange(6), cs_1, label = 'Исходные данные')
    plt.title(label = title)
    plt.legend()
    plt.savefig(save)
    plt.show()


plt_fit(cs_en, cs_ru, popt_en, popt_ru, title = "Предсказание для иностранного ПО", save = 'foreign_pred.png')
plt_fit(cs_ru, cs_en, popt_ru, popt_en, title = "Предсказание для отечественного ПО", save = 'domestic_pred.png')


cs_fit_en = fit_model(cs_en, cs_ru, popt_en, popt_ru)[0]
cs_fit_ru = fit_model(cs_ru, cs_en, popt_ru, popt_en)[0]


# Оценка относительной ошибки аппроксимации линейной модели
def rae(cs_fit, cs):
    abslte_error = np.sum(np.abs(cs - cs_fit))
    cs_mean = np.mean(cs)
    abslte_diff = np.sum(np.abs(cs - cs_mean))
    rae = abslte_error / abslte_diff

    return rae

print("Относительная абсолютная ошибка для аппроксимации данных продаж иностанного ПО =", rae(cs_fit_en[3:], cs_en[4:]))
print("Относительная абсолютная ошибка для аппроксимации данных продаж отечественного ПО =", rae(cs_fit_ru[3:], cs_ru[4:]))


# Предсказание линейной модели без субсидии
prediction_en, prediction_ru = fit_model(cs_en, cs_ru, popt_en, popt_ru, t = 15)
# Цель субсидии Q - увеличение продаж в 2030 году на 30 % в сравнении с прогнозом
q = 1.30 * prediction_ru[-1]
print("Цель субсидии Q =", q)

# Определение субсидии

# Есть цель Q, нужна ступенчатая функция для s.
sub_start_ru = prediction_ru[4]
sub_start_en = prediction_en[4]
t = 10


def calc_x(start_ru, start_en, s_t):
    a1, b1, y1 = popt_ru
    a2, b2, y2 = popt_en
    x1_i = a1 - b1 * (n1 - start_ru - s_t) + y1 * (n2 - start_en)
    x2_i = a2 - b2 * (n2 - start_en) + y2 * (n1 - start_ru - s_t)
    return x1_i, x2_i


def objective(s):
    start_ru = sub_start_ru
    start_en = sub_start_en
    a1, b1, y1 = popt_ru
    a2, b2, y2 = popt_en
    x1 = []
    x2 = []
    s_t = s[0]
    for i in range(t):
        if i == 4:
            s_t = s[1]
        elif i == 7:
            s_t = s[2]
        x1_i, x2_i = calc_x(start_ru, start_en, s_t)
        x1.append(x1_i)
        x2.append(x2_i)
        start_ru = x1_i
        start_en = x2_i
    return x1[-1]


# Ограничение - результат должен быть равен Q 
def constr1(s):
    return objective(s) - q


con1 = {'type': 'eq', 'fun': constr1}

# Процесс оптимизации
methods_yak = ['Newton-CG', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov']
methods = ['Nelder-Mead', 'Powell', 'CG', 'BFGS', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP', 'trust-constr']
b1 = (0, 800)
b2 = (0, 800)
b3 = (0, 800)
bnds = (b1, b2, b3)
x0 = np.zeros(3) # предположительные значения субсидии s1, s2, s3
x0[0], x0[1], x0[2] = 5000, 3500, 2000
sol = minimize(objective, x0 = x0, method = methods[8], bounds = bnds, constraints = con1)
print(sol) # Печатаем результат с оценкой работы оптимизатора


# Вычисление погодового прогноза с учетом субсидии
def sub_model(s):
    start_ru = sub_start_ru
    start_en = sub_start_en
    a1, b1, y1 = popt_ru
    a2, b2, y2 = popt_en
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


# Сравнение изначального прогноза и результата субсидии
plt.plot(np.arange(5, 15), sub_model(sol.x)[0], 'r',  label = 'Выходные данные модели с учетом субсидии')
plt.plot(np.arange(len(prediction_ru)), prediction_ru, label = 'Выходные данные модели без учета субсидии')
plt.plot(len(prediction_ru) - 1, q, marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green", label = "Цель субсидии Q") 
plt.title(label = 'Сравнение изначального прогноза и результата субсидии')
plt.legend()
plt.savefig('sub_res.png')
plt.show()

# Subsidy step change plot
plt.step(np.arange(10), sub_model(sol.x)[1])
plt.title(label = "Измение размера субсидии в период ее действия")
plt.savefig('sub_step.png')
plt.show()
