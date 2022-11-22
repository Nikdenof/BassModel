import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


table_name = "SAPR.xlsx"
table = pd.read_excel(table_name, skiprows = 2, header = None)
table = table.transpose()[1:]
table.columns = ["Foreign", "Domestic", "Sum"]
table["Years"] = table.index

cad_en = table[["Years", "Foreign"]].copy()
cad_ru = table[["Years", "Domestic"]].copy()
cad_sum = table[["Years", "Sum"]].copy()


def cum_sum_add(df):
    df = df.rename(columns={ df.columns[1]: "revenues" })
    df['cum_sum'] = df['revenues'].cumsum()
    return df


cad_en = cum_sum_add(cad_en)
cad_ru = cum_sum_add(cad_ru)
cad_sum = cum_sum_add(cad_sum)


def c_t(x, p, q, m):
    return (p+(q/m)*(x))*(m-x)


len_table = int(table.shape[0])
pt_en, pc_en = curve_fit(c_t, cad_en.cum_sum.iloc[0:5],
                         cad_en.revenues.iloc[1:6])
popt_ru, pcov_ru = curve_fit(c_t, cad_ru.cum_sum.iloc[0:5],
                             cad_ru.revenues.iloc[1:6], maxfev=5000)


def bass_model(p, q, m, T = 15):
    Y = [0]
    S = []
    for t in range(T):
        s = p * m + (q - p) * Y[t] - (q / m) * Y[t] ** 2
        S.append(s)
        y = Y[t] + s
        Y.append(y)
    return S, np.cumsum(S) 
        

S_ru, CS_ru = bass_model(*popt_ru)
S_en, CS_en = bass_model(*pt_en)


def rel_plot(): #delete to run
    plt.plot(cad_en['Years'], cad_en['revenues'], label = 'Иностранная продукция')
    plt.plot(cad_ru['Years'], cad_ru['revenues'], label = 'Отечественная продукция')
    plt.plot(cad_sum['Years'], cad_sum['revenues'], label = 'Суммарные продажи')
    plt.title('Продажи ПО')
    plt.legend(loc = 'best')
    plt.show()


def sum_plot(): #delete to run
    plt.plot(cad_en['Years'], cad_en['cum_sum'], label = 'Иностранная продукция')
    plt.plot(cad_ru['Years'], cad_ru['cum_sum'], label = 'Отечественная продукция')
    plt.plot(cad_sum['Years'], cad_sum['cum_sum'], label = 'Суммарные продажи')
    plt.title('Продажи ПО суммарные')
    plt.legend(loc = 'best')
    plt.show()


# rev_plot but for fit data
def fit_plot(data, S):
    year_start = 2016
    year_end = 2016 + len(S)
    plt.plot(data['Years'] + year_start - 1, data['revenues'], label = "Реальные данные")
    plt.plot(np.arange(year_start, year_end), S, label = "Предсказанные данные")
    plt.title('Сравнение прогноза и данных')
    plt.legend(loc = 'best')
    plt.show()

#Uncomment to run
fit_plot(cad_ru, S_ru)
fit_plot(cad_en, S_en)


def cum_fit_plot(data, CS):
    year_start = 2016
    year_end = 2016 + len(CS)
    plt.plot(data['Years'] + year_start - 1, data['cum_sum'], label = "Реальные данные")
    plt.plot(np.arange(year_start, year_end), CS, label = "Предсказанные данные")
    plt.title('Сравнение кумулитивных данных и их прогноза')
    plt.legend(loc = 'best')
    plt.show()


#Uncomment to run
cum_fit_plot(cad_ru, CS_ru)
cum_fit_plot(cad_en, CS_en)

#Вывод переменных
print('p,q,m для иностранного ПО', pt_en)
print('p,q,m для отечественного ПО', popt_ru)

#Результаты аппроксимации
print('Аппроксимация продаж для иностранного ПО \n', S_en)
print('Аппроксимация продаж для отечественного ПО \n', S_ru)
