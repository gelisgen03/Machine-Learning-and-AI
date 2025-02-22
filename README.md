# Machine-Learning-and-AI

# IMDB Puan Tahmini Projesi 🎬

* ⏺️ Bu proje, filmlerin çeşitli özelliklerini kullanarak IMDB puanlarını tahmin etmek için makine öğrenmesi tekniklerini içeren bir uygulamadır. Proje, Python ve popüler veri bilimi kütüphaneleri kullanılarak geliştirilmiştir.

## Proje Özellikleri ℹ️
Aşağıda, proje kapsamında kullanılan veri setinin sütunları ve açıklamaları bulunmaktadır:

1. **id**: Her film için benzersiz bir kimlik numarası.
2. **title**: Filmin adı.
3. **vote_average**: Kullanıcılardan alınan ortalama puan (0 ile 10 arasında bir ölçekte).
4. **vote_count**: Filmin aldığı toplam oy sayısı.
5. **status**: Filmin mevcut durumu ("Yayında," "Yapım Aşamasında" gibi).
6. **release_date**: Filmin resmi olarak yayınlandığı tarih.
7. **revenue**: Filmin elde ettiği toplam gelir (genellikle USD cinsinden).
8. **runtime**: Filmin süresi (dakika cinsinden).
9. **adult**: Filmin yetişkinlere yönelik olup olmadığını belirten bir ifade ("True" veya "False").
10. **budget**: Filmin yapım maliyeti (genellikle USD cinsinden).
11. **imdb_id**: IMDb üzerindeki benzersiz film kimliği.
12. **original_language**: Filmin orijinal dili (örneğin, "en" İngilizce için).
13. **original_title**: Filmin orijinal dilindeki adı.
14. **overview**: Filmin konusuna dair kısa bir özet.
15. **popularity**: Filmin popülerliğini gösteren bir metrik (genellikle görüntülenme veya arama verileriyle ilişkilidir).
16. **tagline**: Filmle ilişkilendirilen kısa bir slogan veya ifade.
17. **genres**: Filmin ait olduğu türler (örneğin, Aksiyon, Komedi, Drama).
18. **production_companies**: Filmin yapımında yer alan şirketlerin adları.
19. **production_countries**: Filmin yapıldığı ülkeler.
20. **spoken_languages**: Filmde konuşulan diller.
21. **keywords**: Filmle ilişkilendirilen önemli terimler veya kategoriler.

---

## Proje Adımları 🪜
Projenin gelişim süreci aşağıdaki adımlarla gerçekleştirilmiştir:
* Veri seti, bir CSV dosyasından Python ortamına aktarılmıştır. Özellikle eksik veya yanlış değerler kontrol edilmiştir.
### 📥 1. Dataset'ten Veri Çekme 
Kod Parçacığı:
```python
import pandas as pd

data=pd.read_csv('Kitap1.csv')


```


### 🔢 2. Veri Analizi 
* Bu aşamada, verinin yapısı incelenmiş ve özelliklerin istatistiksel özeti elde edilmiştir. Hangi özelliklerin hedef değişkeni (vote_average) tahmin etmekte faydalı olabileceği belirlenmiş ve gereksiz özellikler dataset'ten kaldırılmıştır.

Dataset Analizi:
```python
data.head()
data.dtypes
data.info()
data.isnull().sum()
```

Tespit Edilen Gereksiz Özelliklerin Kaldırılması:
```python
data=data.drop(['id','title','keywords','status','original_title','overview','tagline','spoken_languages','imdb_id'],axis=1)
```
![datahead](https://github.com/user-attachments/assets/acd5cd80-3196-4164-905b-e106394e6098)
![datainfo](https://github.com/user-attachments/assets/57296f8e-42fe-46ed-8c06-18b99ce83291)
![dataisnull](https://github.com/user-attachments/assets/c5341ad4-3ced-4c49-8485-a45832838fcb)



### 🌟 3. Dataset'in Encoding'e Hazırlanması 
* Bu aşamada verilerin geniş çapta işlenmesi gerçekleştirilmiştir. Eksik değerler doldurulmuştur.

➡️ 'original_language' özelliği için örnek uygulama:
```python
#original_language değişkeninde toplam sayısı 12 ten az olan dillerin satırlarını kaldır
#(12 sayısı optimum değer olarak belirlenmiştir)
language_counts = data['original_language'].value_counts()
languages_to_keep = language_counts[language_counts >= 12].index
data = data[data['original_language'].isin(languages_to_keep)]
```
➡️ Null Değerler içeren Satırların Kaldırılması:
```python
data = data.dropna(subset=['genres'])
data = data.dropna(subset=['production_companies'])
data = data.dropna(subset=['production_countries'])
```
➡️ Birden Fazla Strink Value içeren özelliklerin düzenlenmesi:
```python
# 'production_countries' sütunundaki değerleri virgüle göre ayır ve uzunluklarını hesapla
data['country_count'] = data['production_countries'].str.split(', ').str.len()

# Ülke sayısı 3'den fazla olan satırları filtre
data = data[data['country_count'] <= 10]

# 'country_count' sütununu sil (artık ihtiyacımız yok)
data = data.drop('country_count', axis=1)
```
➡️ Birden Fazla Strink Value içeren özelliklerin Encoding e hazırlanması:
```python
data['genres_split'] = data['genres'].str.split(', ')
data['production_companies_split'] = data['production_companies'].str.split(', ')
data['production_countries_split'] = data['production_countries'].str.split(', ')
#örnek dönüşüm formatı: Action,Horror,sc-fi --->[Action,Horror,Sc-fi]
```
➡️ 'revenue' özelliği için 0 içeren değerlerin atılması:
```python
zero_revenue_count = data[data['revenue'] == 0]['revenue'].count()
zero_revenue_count
data = data[data['revenue'] != 0]
```
➡️ 'release_date' özelliğinden sadece yıl değerini çekme:
```python
# 'release_date' sütununu datetime nesnesine dönüştürün
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce')

# Yıl bilgisini çıkarın ve yeni bir sütuna ekle
data['release_year'] = data['release_date'].dt.year

# Sonucu görüntüle
data['release_year']
```
#### 📊 Encoding Öncesi Grafikler:
1) Verilerin Birbiri ile karşılaştırılması
![noktaGrafikleri](https://github.com/user-attachments/assets/84d7b84f-9e97-4a29-92f6-72f5feea013e)
2) Encoding Öncesi Korolasyon:
![encodeOnceKorolasyon](https://github.com/user-attachments/assets/a573ec5b-90ad-41a2-8d90-44a87e6bc980)

### 🚀 4. Dataset'e Encoding uygulanması
* Bu aşamada Gerekli özellikler (Kategorik veriler) sayısal formata dönüştürülmüştür ve standardize işlemleri gerçekleştirilmiştir.
➡️ 'adult_encoded' özelliği ni booldan int değere dönüştürme:
```python
data['adult_encoded'] = data['adult'].astype(int)
data.info()
```
➡️ 'original_language' özelliğini sayısal ifadeye dönüştürme (BinarEncoding):
```python
from category_encoders import BinaryEncoder
encoder = BinaryEncoder(cols=['original_language'])
data_encoded = encoder.fit_transform(data)
```
![orginalLanguage](https://github.com/user-attachments/assets/3bda4269-7318-4d46-ada7-f625bd52bf1b)
Birden Fazla Strink Value içeren özellikleri int değere dönüştürme (OneHotEncoding):
```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
encoded_genres = mlb.fit_transform(data['genres_split'])

# Kodlanmış türleri DataFrame'e dönüştürün
encoded_genres_df = pd.DataFrame(encoded_genres, columns=mlb.classes_)

# Kodlanmış türleri orijinal DataFrame'e ekleyin
data_encoded = pd.concat([data, encoded_genres_df], axis=1)
```
![turler](https://github.com/user-attachments/assets/8e84d258-27a3-465e-814a-47fcb0efdde6)

➡️ Büyük değer içeren 'revenue' ve 'vote_count' özelliklerinin Standardize edilmesi:
```python
# Standardize the 'revenue' ve 'vote_count' variable
scaler = StandardScaler()
data_encoded['revenue'] = scaler.fit_transform(data_encoded[['revenue']])
data_encoded['vote_count'] = scaler.fit_transform(data_encoded[['vote_count']])
```
#### 📈 Encoding Sonrası Grafikler:
1) Encoding Sonrası Korolasyon:
![encodeSonraKorolasyon](https://github.com/user-attachments/assets/eab6b546-c157-4afc-895f-c43042305378)
2) Encoding Sonrası BoxPlot'lar:
![averageBox1](https://github.com/user-attachments/assets/5153139b-ef0b-455c-a635-85a2210d6919)
![popularityBox2](https://github.com/user-attachments/assets/0f0354e7-7741-43c6-8d77-69a6e10c31fc)
![YearBox3](https://github.com/user-attachments/assets/d24757ff-387a-410b-a1fa-cc3d6b44a454)
![RevenueBox4](https://github.com/user-attachments/assets/975f119b-9212-4caa-9d9c-51f26a4eaf6f)


### 🛑 5. Model Eğitimi ve Model Sonuçları
* Bu aşamada Dört farklı model denenmiş ve en iyi performans Random Forest Regressor ile elde edilmiştir. İstatistiksel metriklerle modeller aşağıda karşılaştırılmıştır.
🔼 Train-Test Ayırma:
```python
# 1. Define features (X) and target (y)
X = data_encoded.drop(['vote_average'], axis=1)
y = data_encoded['vote_average']

# 2. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
➡️ Random Forest:
```python
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Adjust hyperparameters as needed
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
➡️ Linear Regression:
```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)
```
➡️ Logistic Regression:
```python
threshold = data_encoded['vote_average'].median()
data_encoded['vote_average_binary'] = (data_encoded['vote_average'] > threshold).astype(int)
X_logreg = data_encoded.drop(['vote_average', 'vote_average_binary'], axis=1)
y_logreg = data_encoded['vote_average_binary']
X_train_logreg, X_test_logreg, y_train_logreg, y_test_logreg = train_test_split(
    X_logreg, y_logreg, test_size=0.33, random_state=42
)
logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train_logreg, y_train_logreg)
logistic_pred = logistic_model.predict(X_test_logreg)
```
➡️ KMeans Regression:
```python
_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
kmeans.fit(X_train)
train_clusters = kmeans.labels_
kmeans_models = {}
for cluster in range(n_clusters):
    cluster_data_indices = train_clusters == cluster
    X_cluster = X_train[cluster_data_indices]
    y_cluster = y_train[cluster_data_indices]
    model = LinearRegression()
    model.fit(X_cluster, y_cluster)
    kmeans_models[cluster] = model
test_clusters = kmeans.predict(X_test)
kmeans_pred = [kmeans_models[cluster].predict(X_test.iloc[[i]])[0] for i, cluster in enumerate(test_clusters)]
```

### 💬 6. Model Sonuçlarının Karşılaştırılması ve Yorumlanması

* Model Sonuçları
  ![SonuclarTablo1](https://github.com/user-attachments/assets/e94cc855-62df-459e-8429-119df3c5d875)
  

| Model                 | Ortalama Kare Hatası (MSE) | R-Kare Skoru |
|-----------------------|--------------------------|---------------|
| Random Forest         | En düşük               | En yüksek    |
| Linear Regression     | Orta                    | Orta          |
| KMeans Regression     | En yüksek               | En düşük    |
| Logistic Regression   | -                       | -             |

- **Random Forest Regressor** en iyi performansı göstermiştir.
- **Linear Regression** temel bir model olarak iyi bir performans sergilemiştir.
- **KMeans Regression** bu veri setinde düşük bir performans sergilemiştir.
- **Logistic Regression**, sınıflandırma problemleri için uygundur ve bu veri setinde değerlendirilmemiştir.
  
#### Grafikler:
![DegerlendirmeLine1](https://github.com/user-attachments/assets/8ec24d61-912a-427d-a0f8-bf2dbe96a923)
![DegerlendirmeBar1](https://github.com/user-attachments/assets/4ede1e92-45c9-42ab-a849-b649532460b3)



---

## Projenin Amacı 🗒️
Bu proje, makine öğrenmesi algoritmalarının tahmin problemlerindeki gücünü anlamaya yönelik bir çalışmadır. Elde edilen sonuçlar, farklı modellerin veri setine uyumunu karşılaştırma fırsatı sunar.

## Gereksinimler 💥
- Python 3.8+
- Pandas, NumPy, Scikit-learn

Kurulum:
```bash
pip install -r requirements.txt
```
## Tanıtım Videosu 📺
https://youtu.be/-BRt86qTgYo
## Sertifikalar 📄
--> Python Eğitimi: [Python_ile_Makine_Öğrenmesi_Sertifika.pdf](https://github.com/user-attachments/files/18470156/Python_ile_Makine_Ogrenmesi_Sertifika.pdf)

