import numpy as np

# Оценка относительной ошибки аппроксимации линейной модели
def rae(cs_fit, cs):
    abslte_error = np.sum(np.abs(cs - cs_fit))
    cs_mean = np.mean(cs)
    abslte_diff = np.sum(np.abs(cs - cs_mean))
    rae = abslte_error / abslte_diff

    return rae
