"""
"""

import numpy as np
import matplotlib.pyplot as plt


def fit_plt(cs, model_output, title, save):
    plt.plot(np.arange(3, len(cs)), model_output[2: len(cs)-1], 'r',  label = 'Предсказания модели')
    plt.plot(np.arange(1, 4), model_output[:3], 'g--',  label = 'Аппроксимация модели')
    plt.plot(np.arange(2), [cs[0], model_output[0]], 'g--')
    plt.scatter(np.arange(6), cs, label = 'Исходные данные')
    plt.title(label = title)
    plt.legend()
    plt.savefig(save)
    plt.show()
