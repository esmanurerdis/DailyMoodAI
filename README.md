# 📊 Enerji Tüketimi Tahmin Projesi (PJME Hourly)

Bu proje, **ABD PJM bölgesine ait saatlik enerji tüketimi verilerini** kullanarak **gelecek enerji talebini tahmin eden** bir makine öğrenmesi uygulamasıdır.  
Veri analizi, özellik mühendisliği, model eğitimi, hiperparametre optimizasyonu ve Türkçe raporlama adımlarını kapsamaktadır.

---

## 🚀 Proje Özellikleri
- **Veri Ön İşleme & EDA**: Eksik verilerin analizi, zaman serisi istatistikleri, görselleştirme
- **Özellik Mühendisliği**: Zaman temelli, lag ve rolling özellikler
- **Modelleme**: Linear Regression ve XGBoost
- **Model Doğrulama**: K-Fold cross-validation
- **Hiperparametre Optimizasyonu**: GridSearchCV ile XGBoost parametre araması
- **Tahmin**: Gelecek 7 gün için saatlik enerji talebi tahmini
- **Raporlama**: Türkçe metrikler ve görseller ile rapor oluşturma

---

## 📂 Proje Yapısı
```bash
energy_project/
├── data/
│   └── PJME_hourly.csv           # Ham veri seti
├── models/
│   └── xgb_pjme_best.pkl         # Eğitilmiş XGBoost modeli
├── outputs/
│   ├── tahminler.csv             # Tahmin sonuçları
│   ├── rapor_gercek_vs_tahmin_*.png
│   ├── rapor_hata_zaman_*.png
│   ├── rapor_hata_hist_*.png
├── main.py                       # Model eğitimi & tahmin üretimi
├── model_validation.py           # Cross-validation
├── hyperparam_tuning.py          # Hiperparametre optimizasyonu
├── report.py                     # Rapor oluşturma
├── requirements.txt              # Bağımlılık listesi
└── README.md                     # Proje açıklaması
```

---

## 📊 Kullanılan Kütüphaneler
- Python 3.9+
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost
- joblib

---

## 🔧 Kurulum
```bash
# Bağımlılıkları yükle
pip install -r requirements.txt

# Modeli eğit ve 7 günlük tahmini üret
python main.py

# Cross-validation ile doğrulama
python model_validation.py

# Hiperparametre optimizasyonu (opsiyonel)
python hyperparam_tuning.py

# Türkçe rapor üret
python report.py
```

---

## 📊 Veri Seti
- **Kaynak:** [Kaggle - PJME Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- **Kapsam:** 2002–2018 yılları arasında PJM bölgesinde saatlik enerji tüketimi (MW cinsinden)

---

## 💡 Notlar
- Proje, hem **Data Science** hem de **Machine Learning Engineer** portföyü için uygundur.
- Daha yüksek performans için farklı modeller (LightGBM, Prophet vb.) denenebilir.
- Tahminler, enerji planlaması ve talep yönetimi gibi alanlarda kullanılabilir.

---

## 🏷 Lisans
Bu proje MIT Lisansı ile sunulmaktadır.  
Dilediğiniz gibi kullanabilir ve geliştirebilirsiniz.
