import numpy as np
from scipy.optimize import curve_fit 
from utils import rae
from plots import fit_plt
from model_bass import BassModel

sum_foreign, cumsum_foreign= np.genfromtxt('outputs/sales_en.csv', delimiter=',') 
sum_russian, cumsum_russian = np.genfromtxt('outputs/sales_ru.csv', delimiter=',') 

bass_russian = BassModel(cumsum_russian, sum_russian, cumsum_foreign, sum_foreign)

coef_m, coef_s, coef_d = bass_russian.fit(visualize=True)


