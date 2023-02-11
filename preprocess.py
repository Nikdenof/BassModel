import pandas as pd
import numpy as np


# Загрузка начальных данных
table_name = "SAPR.xlsx"
table = pd.read_excel(table_name, skiprows = 2, header = None)
table = table.transpose()[1:]
table.columns = ["Foreign", "Domestic", "Sum"]
table["Years"] = table.index

cad_en = table[["Years", "Foreign"]].copy()
cad_ru = table[["Years", "Domestic"]].copy()


# Выделяем столбец кумулятивной суммы
def cumsum_add(df):
    df = df.rename(columns={df.columns[1]: "revenues"})
    df['cum_sum'] = df['revenues'].cumsum()
    arr = df['revenues'].cumsum().values
    return arr


np.savetxt("sales_en.csv", cumsum_add(cad_en), delimiter=",")
np.savetxt("sales_ru.csv", cumsum_add(cad_ru), delimiter=",")
