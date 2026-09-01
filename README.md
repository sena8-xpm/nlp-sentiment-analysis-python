# 💬 NLTK ve Scikit-Learn ile Duygu Analizi (Sentiment Analysis)

Bu proje, müşteri yorumlarını Doğal Dil İşleme (NLP) teknikleriyle analiz ederek metinlerin duygu durumunu (Olumlu / Olumsuz) otomatik olarak sınıflandırmak amacıyla geliştirilmiştir.

## 📌 Kullanılan Teknolojiler
* **Python 3.x**
* **Scikit-Learn** (`TfidfVectorizer`, `MultinomialNB`, `Pipeline`)
* **Pandas** (Metin Verisi Yapılandırma)

## 🛠️ Uygulanan Adımlar
1. **Metin Vektörleştirme:** Ham metin verileri TF-IDF (Term Frequency-Inverse Document Frequency) yöntemiyle sayısal niteliklere dönüştürüldü.
2. **Model Eğitimi:** Metin sınıflandırmada yüksek performans gösteren **Multinomial Naive Bayes** algoritması kullanıldı.
3. **Pipeline Yapısı:** Metin işleme ve tahmin adımları scikit-learn `Pipeline` mimarisi altında birleştirildi.
4. **Tahmin Mekanizması:** Görünmeyen yeni müşteri yorumlarının duygu durumları başarıyla tahmin edildi.

## 🚀 Proje Kodları
Projenin Python kodlarına [buradan](./main.py) ulaşabilirsiniz.
