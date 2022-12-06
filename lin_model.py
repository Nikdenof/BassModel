import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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
def x1_t(X, n1, n2, a, b, y):
    x1, x2 = X # Кумулятивные продажи первого и второго продукта
    return a - b * (n1 - x1) + y * (n2 - x2) 


