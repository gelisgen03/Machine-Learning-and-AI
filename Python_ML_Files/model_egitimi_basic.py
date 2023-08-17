
#Kutuphaneler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


#veri yükleme
data=pd.read_csv('https://bilkav.com/eksikveriler.csv')
pd.DataFrame(data)

#String verileri kategorikleştirme
kategorik=data.iloc[:,0:1] #string olan sutunun once tum elemanlarini aldik sonra bu sutunun hangi sutun oldugunu belirledik
donusum=pd.get_dummies(kategorik) #get_dummies ile bu kategorik verileri true false ye cevirdik
donusum=donusum*1 #true falseyi 1,0 a cevirdik
data.drop('ulke',axis=1,inplace=True)#data dan ulke yi kaldirdim
new_data=pd.concat([donusum,data],axis=1)#iki dataframe mi axis=1 de yani sutunda birlestirdim

#NaN degereleri doldurma (eksik veriler)
new_data.replace(np.nan,np.mean(new_data['yas']),inplace=True)
print(new_data)

Y=new_data['cinsiyet'] #sadece cinsiyeti aldik
X=new_data.drop('cinsiyet',axis=1)#cinsiyet olmadan tüm data
print(Y),print(X)

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0)

sc=StandardScaler() #bu class sayesinde olcekleme islemlerimizi yapabiliriz. (standardize)
X_test=sc.fit_transform(x_test)
X_train=sc.fit_transform(x_train)
Y_test=sc.fit_transform(y_test)
Y_train=sc.fit_transform(y_train)
