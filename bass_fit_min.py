import numpy as np
# from scipy.optimize import curve_fit 
# from utils import rae
from model_bass import BassModel

sum_foreign, cumsum_foreign= np.genfromtxt('outputs/sales_en.csv', delimiter=',') 
sum_russian, cumsum_russian = np.genfromtxt('outputs/sales_ru.csv', delimiter=',') 

bass_russian = BassModel(cumsum_russian, sum_russian, cumsum_foreign, sum_foreign)

lst = bass_russian.fit()
# result = bass_russian.calc_prediction(num_years=7, visualize=True)
# print(result)

