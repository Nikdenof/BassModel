"""
Данный код выполняет обработку файла в формате Excel,
преобразует в формат csv, который можно подгрузить используя
функции numpy или pandas
"""
import pandas as pd
import numpy as np


# Загрузка начальных данных
TABLE_NAME = "SAPR.xlsx"
table = pd.read_excel(TABLE_NAME, skiprows = 2, header = None)
table = table.transpose()[1:]
table.columns = ["Foreign", "Domestic", "Sum"]
table["Years"] = table.index

cad_en = table[["Years", "Foreign"]].copy()
cad_ru = table[["Years", "Domestic"]].copy()


def cumsum_add(df_mod):
    """
    Выделение столбца кумулятивной суммы, 
    замена формата dataframe на numpy array
    """
    df_mod = df_mod.rename(columns={df_mod.columns[1]: "revenues"})
    df_mod['cum_sum'] = df_mod['revenues'].cumsum()
    # arr = df_mod['revenues'].cumsum().values
    arr = df_mod[['revenues', 'cum_sum']].transpose().values
    return arr


np.savetxt("outputs/sales_en.csv", cumsum_add(cad_en), delimiter=",")
np.savetxt("outputs/sales_ru.csv", cumsum_add(cad_ru), delimiter=",")
