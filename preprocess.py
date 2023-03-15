"""
Данный код выполняет обработку файла в формате Excel,
преобразует в формат csv, который можно подгрузить используя
функции numpy или pandas
"""
import os
import pandas as pd
import numpy as np


# Загрузка начальных данных
#RAW_DATA = "new_sapr.xlsx"
RAW_DATA = "SAPR.xlsx"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DIR = "data/processed/"


table = pd.read_excel(RAW_DATA_DIR+RAW_DATA, skiprows = 2, header = None)
table = table.transpose()[1:]
table.columns = ["Foreign", "Domestic", "Sum"]
table["Years"] = table.index

cad_en = table[["Years", "Foreign"]].copy()
cad_ru = table[["Years", "Domestic"]].copy()


def cumsum_add(df_mod: pd.DataFrame) -> np.ndarray:
    """
    Выделение столбца кумулятивной суммы, 
    dataframe в numpy array
    """
    df_mod = df_mod.rename(columns={df_mod.columns[1]: "revenues"})
    df_mod['cum_sum'] = df_mod['revenues'].cumsum()
    # arr = df_mod['revenues'].cumsum().values
    arr = df_mod[['revenues', 'cum_sum']].transpose().values
    return arr


os.makedirs(PROCESSED_DIR, exist_ok=True)
np.savetxt(PROCESSED_DIR+"sales_en.csv", cumsum_add(cad_en), delimiter=",")
np.savetxt(PROCESSED_DIR+"sales_ru.csv", cumsum_add(cad_ru), delimiter=",")
