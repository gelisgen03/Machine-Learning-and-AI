import pandas as pd
import numpy as np


data=pd.read_csv('https://bilkav.com/eksikveriler.csv')
pd.DataFrame(data)
print(data)

print(data['yas'])
d_y_m=np.mean(data['yas'])
data['yas'] = data['yas'].replace(np.nan, d_y_m)
print(data['yas'])