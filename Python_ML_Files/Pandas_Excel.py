import pandas as pd
import numpy as py

d=pd.read_excel('ornek.xlsx')
pd.DataFrame(d)
print(d)
d.dropna(inplace=True)
print(d)
d.to_excel('Nulless.xlsx')
