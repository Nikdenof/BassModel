import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.DataFrame({'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     'revenues': [0.1, 3, 5.2, 7, 5.25, 4.9, 3, 2.4, 1.9, 1.3, 0.8, 0.6]
                     })
data['cum_sum'] = data['revenues'].cumsum()

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


popt, pcov = curve_fit(c_t, data.cum_sum[0:11], data.revenues[1:12])


def rel_plot():
    plt.plot(cad_en['Years'], cad_en['revenues'], label = 'Иностранная продукция')
    plt.plot(cad_ru['Years'], cad_ru['revenues'], label = 'Отечественная продукция')
    plt.plot(cad_sum['Years'], cad_sum['revenues'], label = 'Суммарные продажи')
    plt.title('Продажи ПО')
    plt.legend(loc = 'best')
    plt.show()


def sum_plot():
    plt.plot(cad_en['Years'], cad_en['cum_sum'], label = 'Иностранная продукция')
    plt.plot(cad_ru['Years'], cad_ru['cum_sum'], label = 'Отечественная продукция')
    plt.plot(cad_sum['Years'], cad_sum['cum_sum'], label = 'Суммарные продажи')
    plt.title('Продажи ПО суммарные')
    plt.legend(loc = 'best')
    plt.show()


# rev_plot but for fit data
def fit_plot():
    plt.plot(data['week'], c_t(data['cum_sum'], *popt))
    plt.show()
