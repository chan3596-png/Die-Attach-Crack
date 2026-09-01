import pandas as pd
import json

df = pd.read_excel('iii_die attatch/20000_BGTTV.xlsx', header=None, nrows=40)
data = df.fillna("").values.tolist()
with open('temp_cols.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False)
