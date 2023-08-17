import numpy as np
import pandas as pd
data_url = "https://ndownloader.figshare.com/files/12710618"
yeni_df=pd.DataFrame(pd.read_csv(data_url))

data=np.random.randn(4,4)

df=pd.DataFrame(data,index=['satir1','satir2','satir3','satir4'],columns=['A','B','C','D']) #Label ekledim
print(df)
df.rename(columns={'A':'Asim','B':'Mehmet','C':'Nur','D':'Muradiye'},inplace=True)
print(df)
print(df[['Asim','Nur']]) # ikisini yazdirdim fakat array içinde
df['Emine']=df['Mehmet']*2 #sütün satir ekledim
print(df)
df.drop('Mehmet',axis=1,inplace=True) #sildim, axis=0 satir adlari,axis=1 sutun adlari
print(df)
print('Loc func:\n',df.loc["satir1"]) #satir 1 in verilerini cektim
print('ILock func:\n',df.iloc[2]) # 2 indisli satirin verilerini cektim -satir3-
boolienFrame=df[df>0] # 0 dan buyukleri sadece gosterdik
print(boolienFrame)
boolienFrame2=df[df['Muradiye']>0] #muradiye sutununun sadece 0 dan buyuk satirlarini getirdi
print(boolienFrame2)
boolienFrame3=yeni_df['precip']>1
print(yeni_df)
print(boolienFrame3)
yeni_df['Yeni index']=[1,2,3,4,5,6,7,8,9,10,11,12]
print(yeni_df.set_index('Yeni index')) #yeni index taması
#puf noktalar
print(yeni_df['seasons'].unique())
print(yeni_df['seasons'].nunique()) # kac tane benzersiz var
print(yeni_df['seasons'].value_counts()) # hangisinden kactane