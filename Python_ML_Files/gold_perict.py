import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


df=pd.read_csv('all_commodities_data.csv')
pd.DataFrame(df)
df=df[0:1000]
print(df.isnull().sum()) #kapıp data kontrol
print(df.dtypes) #data türleri

kategorik=df.iloc[:,1:3] 
donusum=pd.get_dummies(kategorik) 
donusum=donusum*1
#print(donusum)

df.drop(['ticker','commodity','date'],axis=1,inplace=True)
#print(df)

new_df=pd.concat([donusum,df],axis=1)

Y=new_df['volume']
X=new_df.drop('volume',axis=1)
#print(X)
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)

x_train=pd.DataFrame(x_train)
x_test=pd.DataFrame(x_test)
y_train=pd.DataFrame(y_train)
y_test=pd.DataFrame(y_test)


sc=StandardScaler()
X_test=sc.fit_transform(x_test)
X_train=sc.fit_transform(x_train)
Y_test=sc.fit_transform(y_test)
Y_train=sc.fit_transform(y_train)
Y_test=pd.DataFrame(Y_test)

lr=LinearRegression()
lr.fit(X_train,Y_train)
tahmin=lr.predict(X_test) #Y_test bulma
lr.fit(x_train,y_train) #without standardize
tahmin2=lr.predict(x_test)
tahmin=pd.DataFrame(tahmin)
tahmin2=pd.DataFrame(tahmin2)

x_train=x_train.sort_index() #indexleri siraladik yoksa nasil karsılastircan, en basta uretirken karistiriilmisti random ile
y_train=y_train.sort_index()
x_test=x_test.sort_index()
Y_test=Y_test.sort_index()
tahmin=tahmin.sort_index()
tahmin2=tahmin2.sort_index()
y_test=y_test.sort_index()



#print(y_test)

