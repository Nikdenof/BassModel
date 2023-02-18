class LinearModel():
    def __init__(self) -> None:
        pass
    def objective_func(self):
        pass
    def calc_x(start_ru, start_en, s_t):
        a1, b1, y1 = coefficients_ru
        a2, b2, y2 = coefficients_en
        x1_i = a1 - b1 * (n1 - start_ru - s_t) + y1 * (n2 - start_en)
        x2_i = a2 - b2 * (n2 - start_en) + y2 * (n1 - start_ru - s_t)
        return x1_i, x2_i
    def calculate_model(self):
        start_ru = sub_start_ru
        start_en = sub_start_en
        a1, b1, y1 = coefficients_ru
        a2, b2, y2 = coefficients_en
        x1 = []
        x2 = []
        s_lst = []
        s_t = s[0]
        for i in range(t):
            if i == 4:
                s_t = s[1]
            elif i == 7:
                s_t = s[2]
            x1_i, x2_i = calc_x(start_ru, start_en, s_t)
            x1.append(x1_i)
            x2.append(x2_i)
            s_lst.append(s_t)
            start_ru = x1_i
            start_en = x2_i
        return x1, s_lst

""" 
I need to also incorporate the minimization into the class
Maybe it is a good idea, maybe not
"""
