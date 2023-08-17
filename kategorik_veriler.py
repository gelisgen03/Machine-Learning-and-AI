import pandas as pd
import numpy as np

data=pd.read_csv('https://bilkav.com/eksikveriler.csv')
pd.DataFrame(data)

kategorik=data.iloc[:,0:1] #string olan sutunun once tum elemanlarini aldik sonra bu sutunun hangi sutun oldugunu belirledik
donusum=pd.get_dummies(kategorik) #get_dummies ile bu kategorik verileri true false ye cevirdik
donusum=donusum*1 #true falseyi 1,0 a cevirdik
data.drop('ulke',axis=1,inplace=True)#data dan ulke yi kaldirdim
new_data=pd.concat([data,donusum],axis=1)#iki dataframe mi axis=1 de yani sutunda birlestirdim
print(new_data)
new_data2=new_data.replace(['e','k'],[0,1]) #erkek ve kadini 0,1 olarak replace ettik
print(new_data2) #son hali bu