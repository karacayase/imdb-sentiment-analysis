# =============================================================================
# 1. TEMEL AYARLAR VE KÜTÜPHANELER
# =============================================================================

MAX_REVIEW_LENGTH = 300   # Bir yorumdan alınacak maksimum kelime (token) sayısı
EMBEDDING_VECTOR_SIZE = 100 # Kelime gömme vektörünün boyutu

import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, Reshape, Dense, Dropout # Dropout eklendi
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# =============================================================================
# 2. VERİ YÜKLEME VE ÖN İŞLEME
# =============================================================================
# Veri setini yükle
veri_seti_df = pd.read_csv('IMDB Dataset.csv')

# CountVectorizer ile kelime dağarcığının (vocabulary) oluşturulması
vektorleyici = CountVectorizer()
vektorleyici.fit(veri_seti_df['review'])
kelime_sayisi = len(vektorleyici.vocabulary_) + 1 # +1, padding (0) için

# Metinleri indeks dizilerine dönüştürme
yorum_indeksleri = [[vektorleyici.vocabulary_[word] + 1
                     for word in re.findall(r'(?u)\b\w\w+\b', metin.lower()) 
                     if word in vektorleyici.vocabulary_] # Sadece bilinen kelimeler
                    for metin in veri_seti_df['review']]

# pad_sequences ile dizileri eşit uzunluğa getirme
X_verisi = pad_sequences(yorum_indeksleri, MAX_REVIEW_LENGTH, dtype='float32')

# Etiketleri (Y) sayısal NumPy dizisine dönüştürme
Y_etiketler = (veri_seti_df['sentiment'] == 'positive').to_numpy(dtype='uint8')

# Eğitim ve test setlerine ayırma
X_train, X_test, Y_train, Y_test = train_test_split(X_verisi, Y_etiketler, test_size=0.25, random_state=42)

# =============================================================================
# 3. DERİN ÖĞRENME MODELİ (Word Embedding ve Dropout ile)
# =============================================================================
model = Sequential(name='Ozgun_IMDB_Sentiment_Analyzer') # Özgün Model Adı
model.add(Input((MAX_REVIEW_LENGTH, ), name='Girdi_Dizisi'))
# Embedding Katmanı: Kelimelerin anlamsal temsillerini öğrenir
model.add(Embedding(kelime_sayisi, EMBEDDING_VECTOR_SIZE, name='Kelime_Gomme'))
# Reshape: 3 boyutlu tensörü 2 boyutlu hale getirir (Dense katmanına uygun)
model.add(Reshape((-1, ), name='Vektor_Duzlestirme'))

# Gizli Katmanlar: Aşırı öğrenmeyi engellemek için Dropout eklendi
model.add(Dense(512, activation='relu', name='Gizli_Katman_1'))
model.add(Dropout(0.4)) # %40 Dropout
model.add(Dense(128, activation='relu', name='Gizli_Katman_2'))
model.add(Dropout(0.2)) # %20 Dropout
model.add(Dense(1, activation='sigmoid', name='Cikti_Duygu_Skoru'))

model.summary()

# Modeli derleme:'adam' optimizasyoncusu kullanıldı
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['binary_accuracy'])

# =============================================================================
# 4. MODEL EĞİTİMİ
# =============================================================================
# Erken Durdurma (Early Stopping)
esc = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

print("\nModel eğitimi başlıyor...")
egitim_gecmisi = model.fit(X_train, Y_train, 
                           epochs=100, 
                           batch_size=64,
                           validation_split=0.2, 
                           callbacks=[esc])

# =============================================================================
# 5. GÖRSELLEŞTİRME VE DEĞERLENDİRME
# =============================================================================
# Kayıp Grafiği
plt.figure(figsize=(14, 6))
plt.title('Duygu Analizi - Eğitim ve Doğrulama Kayıp Eğrileri (Adam Opt.)', pad=10, fontsize=16)
plt.xlabel('Epok Sayısı')
plt.ylabel('Binary Crossentropy Kaybı')
plt.plot(egitim_gecmisi.epoch, egitim_gecmisi.history['loss'], label='Eğitim Kaybı')
plt.plot(egitim_gecmisi.epoch, egitim_gecmisi.history['val_loss'], label='Doğrulama Kaybı')
plt.legend()
plt.grid(True)
plt.show()

# Başarı Grafiği
plt.figure(figsize=(14, 6))
plt.title('Eğitim ve Doğrulama Başarı Grafiği', pad=10, fontsize=16)
plt.xlabel('Epok Sayısı')
plt.ylabel('Başarı Oranı (Binary Accuracy)')
plt.plot(egitim_gecmisi.epoch, egitim_gecmisi.history['binary_accuracy'], label='Eğitim Başarısı')
plt.plot(egitim_gecmisi.epoch, egitim_gecmisi.history['val_binary_accuracy'], label='Doğrulama Başarısı')
plt.legend()
plt.grid(True)
plt.show()

# Test verisi üzerinde değerlendirme
degerlendirme_sonuclari = model.evaluate(X_test, Y_test, verbose=0)
print("\n--- Test Verisi Değerlendirme Sonuçları ---")
for metrik_adi, sonuc in zip(model.metrics_names, degerlendirme_sonuclari):
    print(f'{metrik_adi.capitalize()}: {sonuc:.4f}')

# =============================================================================
# 6. TAHMİN KISMI (predict-imdb.csv dosyası ile)
# =============================================================================
try:
    tahmin_df = pd.read_csv('predict-imdb.csv')
    
    # Yeni veriyi de aynı şekilde sayısal indekslere dönüştürme
    tahmin_yorum_indeksleri = [[vektorleyici.vocabulary_[word] + 1
                                for word in re.findall(r'(?u)\b\w\w+\b', metin.lower())
                                if word in vektorleyici.vocabulary_]
                               for metin in tahmin_df['review']]

    X_tahmin = pad_sequences(tahmin_yorum_indeksleri, MAX_REVIEW_LENGTH, dtype='float32')
    
    tahmin_sonuclari = model.predict(X_tahmin)
    
    print("\n--- Tahmin Sonuçları ---")
    for yorum, skor in zip(tahmin_df['review'], tahmin_sonuclari[:, 0]):
        duygu = 'Pozitif' if skor > 0.5 else 'Negatif'
        print(f'Yorum: "{yorum[:60]}..." => {duygu} (Skor: {skor:.4f})')
except FileNotFoundError:
    print("\nUyarı: 'predict-imdb.csv' dosyası bulunamadığı için tahmin kısmı atlandı.")