
# IMDB Puan Tahmini Projesi

Bu proje, filmlerin çeşitli özelliklerini kullanarak IMDB puanlarını tahmin etmek için makine öğrenmesi tekniklerini içeren bir uygulamadır. Proje, Python ve popüler veri bilimi kütüphaneleri kullanılarak geliştirilmiştir.

## Proje Özellikleri
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

## Proje Adımları
Projenin gelişim süreci aşağıdaki adımlarla gerçekleştirilmiştir:

### 1. Veri Çekme
Kod Parçacığı:
```python
import pandas as pd

data = pd.read_csv('movies_dataset.csv')
print(data.head())
```
Veri seti, bir CSV dosyasından Python ortamına aktarılmıştır. Özellikle eksik veya yanlış değerler kontrol edilmiştir.

### 2. Veri Analizi
Kod Parçacığı:
```python
print(data.info())
print(data.describe())
```
Bu aşamada, verinin yapısı incelenmiş ve özelliklerin istatistiksel özeti elde edilmiştir. Hangi özelliklerin hedef değişkeni (vote_average) tahmin etmekte faydalı olabileceği belirlenmiştir.

### 3. Veri Kategorikleştirme ve İşlenmesi
Kod Parçacığı:
```python
from sklearn.preprocessing import OneHotEncoder

data = data.fillna(0)  # Eksik verilerin doldurulması
encoder = OneHotEncoder()
encoded_genres = encoder.fit_transform(data['genres'].values.reshape(-1, 1)).toarray()
```
Bu aşamada, kategorik veriler sayısal formata dönüştürülmüştür. Eksik değerler doldurulmuş ve gerekirse özelliklerin ölçeklenmesi sağlanmıştır.

### 4. Model Eğitimi ve Model Sonuçları
Kod Parçacığı:
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X = data.drop(columns=['vote_average'])
y = data['vote_average']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
score = model.score(X_test, y_test)
print(f"R-Kare Skoru: {score}")
```
Dört farklı model denenmiş ve en iyi performans Random Forest Regressor ile elde edilmiştir. İstatistiksel metriklerle modeller karşılaştırılmıştır.

### 5. Model Sonuçlarının Karşılaştırılması ve Yorumlanması
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

---

## Projenin Amacı
Bu proje, makine öğrenmesi algoritmalarının tahmin problemlerindeki gücünü anlamaya yönelik bir çalışmadır. Elde edilen sonuçlar, farklı modellerin veri setine uyumunu karşılaştırma fırsatı sunar.

## Gereksinimler
- Python 3.8+
- Pandas, NumPy, Scikit-learn

Kurulum:
```bash
pip install -r requirements.txt
```

## Katkıda Bulunma
Bu proje geliştirilmeye açıktır. Katkıda bulunmak için lütfen bir pull request oluşturun.

## Lisans
MIT Lisansı
