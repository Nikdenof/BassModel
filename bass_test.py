import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

data = pd.DataFrame({'week': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                     'revenues': [0.1, 3, 5.2, 7, 5.25, 4.9, 3, 2.4, 1.9, 1.3, 0.8, 0.6]
                     })
data['cum_sum'] = data['revenues'].cumsum()


def c_t(x, p, q, m):
    return (p+(q/m)*(x))*(m-x)


popt, pcov = curve_fit(c_t, data.cum_sum[0:11], data.revenues[1:12])


def rel_plot():
    data.plot(x = 'week', y = 'revenues')
    plt.show()


def sum_plot():
    data.plot(x = 'week', y = 'cum_sum')
    plt.show()


# rev_plot but for fit data
def fit_plot():
    plt.plot(data['week'], c_t(data['cum_sum'], *popt))
    plt.show()
