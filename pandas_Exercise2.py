import pandas as pd

        
n=1
data_url = "https://ndownloader.figshare.com/files/12710618"

yeni_df=pd.DataFrame(pd.read_csv(data_url))
def ata():
    for n in range(1,len(yeni_df) ):
        return n
    
yeni_df2=pd.DataFrame(pd.read_csv(data_url),index=[1,2,3,4,5,6,7,8,9,10,11,12])


maasDic={'Departman':['Yazilim','IK','HUKUK','SOSYAL','IT'],'Calisan':['Ahmet','Metmet','Ali','Selim','Kasim'],'Maas':[500,400,200,100,50]}
print(maasDic)
df=pd.DataFrame(maasDic)
print(df)
print(df['Calisan'])
print(df.describe())

grup=df.groupby('Departman') #grupby ile gruplandirma 
print(grup.count()) # birsürü kullanilabicek fonksiyon var grup ile

df2=pd.concat([yeni_df,yeni_df2])
print(df2.info)

d=pd.read_excel("ornek.xlsx")
print(d)

