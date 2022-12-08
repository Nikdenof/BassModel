import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# Исходные данные
q = 40000 # Целевой показатель продаж
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
    df = df.rename(columns={ df.columns[1]: "revenues" })
    df['cum_sum'] = df['revenues'].cumsum()
    arr = df['revenues'].cumsum().values
    return df, arr


cad_en, cs_en = cum_sum_add(cad_en)
cad_ru, cs_ru = cum_sum_add(cad_ru)
cad_sum, cs_sum = cum_sum_add(cad_sum)


# Получение коэффициентов линейной модели спроса
def x_t(X, a, b, y):
    x1, x2 = X # Кумулятивные продажи первого и второго продукта
    # return a - b * (n1 - x1) + y * (n2 - x2) 
    return a - b * x1 + y * x2 


popt_en, _ = curve_fit(x_t, (cs_en[0:5], cs_ru[0:5]), cs_en[1:6])
popt_ru, _ = curve_fit(x_t, (cs_ru[0:5], cs_en[0:5]), cs_ru[1:6])


def fit_model(cs_1, cs_2, popt, t = 5):
    cs_fit = 0

    return cs_fit


def plt_fit(cs_1, cs_2, popt, title):
    plt.plot(np.arange(1, 6), x_t((cs_1[0:5], cs_2[0:5]), *popt), 'r',  label = 'Fit linear model output')
    plt.scatter(np.arange(6), cs_1, label = 'Original data')
    plt.title(label = title)
    plt.legend()
    plt.show()


plt_fit(cs_en, cs_ru, popt_en, title = "Предсказание для иностранного ПО")
plt_fit(cs_ru, cs_en, popt_ru, title = "Предсказание для отечественного ПО")

# NOT CORRECT
#___________________________________________________________
cs_fit_en = x_t((cs_en[0:5], cs_ru[0:5]), *popt_en)
cs_fit_ru = x_t((cs_ru[0:5], cs_en[0:5]), *popt_ru)


# Оценка относительной ошибки аппроксимации линейной модели
def rae(cs_fit, cs):
    abslte_error = np.sum(np.abs(cs - cs_fit))
    cs_mean = np.mean(cs)
    abslte_diff = np.sum(np.abs(cs - cs_mean))
    rae = abslte_error / abslte_diff

    return rae

print("Относительная абсолютная ошибка для аппроксимации данных продаж иностанного ПО =", rae(cs_fit_en, cs_en[1:]))
print("Относительная абсолютная ошибка для аппроксимации данных продаж отечественного ПО =", rae(cs_fit_ru, cs_ru[1:]))


# Предсказание линейной модели без субсидии
