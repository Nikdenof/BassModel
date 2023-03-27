import numpy as np
# from scipy.optimize import curve_fit 
# from utils import rae
from model_bass import BassModel

sum_foreign, cumsum_foreign= np.genfromtxt('data/processed/sales_en.csv', delimiter=',') 
sum_russian, cumsum_russian = np.genfromtxt('data/processed/sales_ru.csv', delimiter=',') 

bass_russian = BassModel(cumsum_russian, sum_russian, cumsum_foreign, sum_foreign)

lst = bass_russian.fit(num_iterations=20000)
result = bass_russian.predict(num_years=5, visualize=True)
