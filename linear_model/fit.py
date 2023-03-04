""" 
В данном скрипте происходит определение аппроксимирующей модели,
получение соответствующих аппроксимирующих коэффициентов.
Для определения качества модели рассчитывается значение ошибки на 
данных, которая модель не использовала для определения коэффициентов.
Также строится базовый прогноз на 15 лет, чтобы в дальнейшем использовать его,
для сравнения действия субсидии
"""

import numpy as np
from scipy.optimize import curve_fit 
from utils import rae
from plots import fit_plt


# Начальные данные
MODEL = "Linear"
# MODEL = "Bass"
_, cs_en = np.genfromtxt('outputs/sales_en.csv', delimiter=',') 
_, cs_ru = np.genfromtxt('outputs/sales_ru.csv', delimiter=',') 
n1 = n2 = 65000


# Получение коэффициентов линейной модели спроса
if MODEL == "Linear":
    def x_t(X, a, b, y):
        x1, x2 = X # Кумулятивные продажи первого и второго продукта
        return a - b * (n1 - x1) + y * (n2 - x2) 
elif MODEL == "Bass":
    def x_t(X, a, b, y):
        x1, x2 = X # Кумулятивные продажи первого и второго продукта
        return a - b * (n1 - x1) + y * (n2 - x2) 
else:
    raise Exception("Нет моделей с таким названием")


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


cs_fit_en = fit_model(cs_en, cs_ru, popt_en, popt_ru)[0]
cs_fit_ru = fit_model(cs_ru, cs_en, popt_ru, popt_en)[0]

fit_plt(cs_en, cs_fit_en, title = "Предсказание для иностранного ПО", save = 'outputs/foreign_pred.png')
fit_plt(cs_ru, cs_fit_ru, title = "Предсказание для отечественного ПО", save = 'outputs/domestic_pred.png')

print("Относительная абсолютная ошибка для аппроксимации данных продаж иностанного ПО =", rae(cs_fit_en[3:], cs_en[4:]))
print("Относительная абсолютная ошибка для аппроксимации данных продаж отечественного ПО =", rae(cs_fit_ru[3:], cs_ru[4:]))

# Предсказание линейной модели без субсидии
prediction_en, prediction_ru = fit_model(cs_en, cs_ru, popt_en, popt_ru, t = 15)

np.savetxt("outputs/base_prediction_ru.csv", prediction_ru, delimiter=",")
np.savetxt("outputs/base_prediction_en.csv", prediction_en, delimiter=",")
np.savetxt("outputs/coef_ru.csv", popt_ru, delimiter=",")
np.savetxt("outputs/coef_en.csv", popt_en, delimiter=",")
