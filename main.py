import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# 1. Örnek Türkçe Müşteri Yorumları Veri Seti
veri = {
    'Yorum': [
        "Ürün harika, kargo çok hızlı geldi kesinlikle tavsiye ederim",
        "Kumaş kalitesi çok kötü, hiç beğenmedim iade edeceğim",
        "Fiyatına göre mükemmel bir performans, çok memnun kaldım",
        "Berbat bir hizmet, paket paramparça ulaştı sakın almayın",
        "Çok başarılı bir uygulama, arayüzü son derece kullanışlı",
        "Ürün eksik geldi ve muhatap bulamıyorum, tam bir hayal kırıklığı"
    ],
    'Etiket': ['Olumlu', 'Olumsuz', 'Olumlu', 'Olumsuz', 'Olumlu', 'Olumsuz']
}

df = pd.DataFrame(veri)

# 2. Metin İşleme (TF-IDF Vectorizer) ve Model (Naive Bayes) Pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# 3. Modeli Eğitme
model.fit(df['Yorum'], df['Etiket'])

print("=== NLP DUYGU ANALİZİ MODELİ EĞİTİLDİ ===\n")

# 4. Yeni Yorumlar İçin Duygu Tahmini
yeni_yorumlar = [
    "Kargo 2 günde geldi, paketleme çok özenliydi teşekkürler",
    "Ürün hemen bozuldu, verdiğim paraya yazık"
]

tahminler = model.predict(yeni_yorumlar)

for yorum, tahmin in zip(yeni_yorumlar, tahminler):
    emoji = "🔴" if tahmin == 'Olumsuz' else "🟢"
    print(f"Yorum: '{yorum}'")
    print(f"Tahmini Duygu: {emoji} {tahmin}\n")
