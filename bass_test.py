import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Начальные данные
sub_year = 2022 # Год начала субсидирования
start_ru_price = 65000 # Начальная цена российского ПО
sub_value = 13000 # Размер субсидии
ru_price = start_ru_price - sub_value # Конечная цена российского ПО после субсидии
en_price = 65000 # Начальная и конечная цена иностранного ПО 
avg_price_start = (start_ru_price + en_price) / 2 # Начальная средняя цена ПО
avg_price_end = (ru_price + en_price) / 2 # Конечная средняя цена ПО

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
    return df


cad_en = cum_sum_add(cad_en)
cad_ru = cum_sum_add(cad_ru)
cad_sum = cum_sum_add(cad_sum)


# Получение коэффициентов модели Басса
def c_t(x, p, q, m):
    return (p+(q/m)*(x))*(m-x)


len_table = int(table.shape[0])
pt_en, pc_en = curve_fit(c_t, cad_en.cum_sum.iloc[0:5],
                         cad_en.revenues.iloc[1:6])
popt_ru, pcov_ru = curve_fit(c_t, cad_ru.cum_sum.iloc[0:5],
                             cad_ru.revenues.iloc[1:6], maxfev=5000)


# Определение базовой модели Басса
def bass_model(p, q, m, T = 15):
    Y = [0]
    S = []
    for t in range(T):
        s = p * m + (q - p) * Y[t] - (q / m) * Y[t] ** 2
        S.append(s)
        y = Y[t] + s
        Y.append(y)
    return np.array(S), np.cumsum(S) 


def sub_bass(p, q, m, alpha,  avgprice_start, avgprice_end, price_start, price_final, T = 15, sub_start = 6):
    Y = [0]
    S = []
    for t in range(sub_start):
        s = p * m + (q - p) * Y[t] - (q / m) * Y[t] ** 2 + alpha * Y[t] * (avgprice_start - price_start)
        S.append(s)
        y = Y[t] + s
        Y.append(y)
    for t in range(sub_start, T):
        s = p * m + (q - p) * Y[t] - (q / m) * Y[t] ** 2 + alpha * Y[t] * (avgprice_end - price_final)
        S.append(s)
        y = Y[t] + s
        Y.append(y)
    return np.array(S), np.cumsum(S) 


# Басс стандартный
S_ru, CS_ru = bass_model(*popt_ru)
S_en, CS_en = bass_model(*pt_en)

# Басс с условием конкуренции
alpha = 0.000005
S_sub_ru, CS_sub_ru = sub_bass(*popt_ru, alpha, avg_price_start, avg_price_end, start_ru_price, ru_price)
S_sub_en, CS_sub_en = sub_bass(*pt_en, alpha, avg_price_start, avg_price_end, en_price, en_price)


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


# Uncomment to run
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


# Uncomment to run
cum_fit_plot(cad_ru, CS_ru)
cum_fit_plot(cad_en, CS_en)


def sub_plot(S_sub, S):
    year_start = 2016
    year_end = 2016 + len(S)
    plt.plot(np.arange(year_start, year_end), S, label = "Предсказанные данные")
    plt.plot(np.arange(year_start, year_end), S_sub, label = "Предсказанные данные с субсидией")
    plt.axvline(x = 2021, color = 'b')
    plt.text(2022, 1000, "Начало действия субсидии", rotation = 90, fontsize = 10)
    plt.title('Сравнение прогноза и данных')
    plt.legend(loc = 'best')
    plt.show()


sub_plot(S_sub_ru, S_ru)
sub_plot(S_sub_en, S_en)


def sub_cum_plot(CS_sub1, CS1, CS_sub2, CS2):
    year_start = 2016
    year_end = 2016 + len(CS_sub1)
    plt.plot(np.arange(year_start, year_end), CS1, label = "Предсказанные данные отечественное ПО")
    plt.plot(np.arange(year_start, year_end), CS_sub1, label = "Предсказанные данные отечественного ПО с субсидией")
    plt.plot(np.arange(year_start, year_end), CS2, label = "Предсказанные данные иностранного ПО")
    plt.plot(np.arange(year_start, year_end), CS_sub2, label = "Предсказанные данные иностранного ПО с субсидией")
    plt.axvline(x = 2021, color = 'b')
    plt.text(2022, 1000, "Начало действия субсидии", rotation = 90, fontsize = 10)
    plt.title('Эффект субсидии на кумулятивные продажи ПО')
    plt.legend(loc = 'best')
    plt.show()
    

sub_cum_plot(CS_sub_ru, CS_ru, CS_sub_en, CS_en)


# MASE -Средняя абсолютная масштабированная ошибка 
def mase(S_fit, df):
    S_real = df['revenues'].to_numpy()
    forecast_error = np.mean(np.abs(S_real - S_fit))
    naive_forecast = np.mean(np.abs(np.diff(S_real)))
    mase = forecast_error - naive_forecast
    
    return mase


def rae(S_fit, df):
    S_real = df['revenues'].to_numpy()
    abslte_error = np.sum(np.abs(S_real - S_fit))
    S_mean = np.mean(S_real)
    abslte_diff = np.sum(np.abs(S_real - S_mean))
    rae = abslte_error / abslte_diff

    return rae



# Вывод переменных
print('p,q,m для иностранного ПО', pt_en)
print('p,q,m для отечественного ПО', popt_ru)

# Результаты аппроксимации
print('Аппроксимация продаж для иностранного ПО \n', S_en)
print('Аппроксимация продаж для отечественного ПО \n', S_ru)

# Оценка ошибки
S_ru_error, CS_ru_error = bass_model(*popt_ru, T = 6)
S_en_error, CS_en_error = bass_model(*pt_en, T = 6)
print('Средняя абсолютная масштабированная ошибка для прогноза продаж отечественногo ПО', mase(S_ru_error, cad_ru))
print('Средняя абсолютная масштабированная ошибка для прогноза продаж иностранного  ПО', mase(S_en_error, cad_en))
print('Относительная абсолютная ошибка для прогноза продаж отечественногo ПО', rae(S_ru_error, cad_ru))
print('Относительная абсолютная ошибка для прогноза продаж иностранного  ПО', rae(S_en_error, cad_en))
