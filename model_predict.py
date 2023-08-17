import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df=pd.read_csv('https://bilkav.com/satislar.csv')
df=pd.DataFrame(df)
print(df.isnull().sum()) #kapıp data kontrol
print(df.dtypes) #data türleri
#print(df)
X=df['Aylar']
Y=df['Satislar']

x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.33,random_state=0) #parametreler=(bagimsiz degisken,bagimli degisken,test_size=test yuzdesi,random)
sc=StandardScaler()

#Reshape etme
'''
x_train=np.reshape(x_train,(-1,1))
x_test=np.reshape(x_test,(-1,1))
y_train=np.reshape(y_train,(-1,1))
y_test=np.reshape(y_test,(-1,1))

'''

#verilerin dataframede olmasi daha kullanişli ve gerekli
x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)

X_train=sc.fit_transform(x_train)
X_test=sc.fit_transform(x_test)
Y_train=sc.fit_transform(y_train)
Y_test=sc.fit_transform(y_test)



#model insaasi (lineer regresyon ile)
lr=LinearRegression()
lr.fit(X_train,Y_train) #X train den Y train i öğrendi
tahmin=lr.predict(X_test) #X testten Y predicti tahmin edecek,sonra bizde bu predict değerini gerçek veri olan Y test ile karşılaştırıcaz
print('Standardize Tahmin\n',tahmin)
print('Standatdize Gercek\n',Y_test)
lr.fit(x_train,y_train)
tahmin2=lr.predict(x_test)
print('Normal Tahmin\n',tahmin2)
print('Normal Gercek\n',y_test)

x_train=x_train.sort_index() #indexleri siraladik yoksa nasil karsılastircan, en basta uretirken karistiriilmisti random ile
y_train=y_train.sort_index()


plt.plot(x_train,y_train)
plt.xlabel('x train')
plt.ylabel('y train')
plt.plot(x_test,lr.predict(x_test)) # dogrusal regresyon ile predict ettiğimiz için doğrusal bir grafi

plt.show()


