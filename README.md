

# 🌟 DailyMoodAI - Multilingual Emotional Intelligence Assistant

![DailyMoodAI Demo](reports/demo_view.png)

**DailyMoodAI** is an AI-powered assistant that breaks language barriers to understand how you feel. It translates your input from any supported language, analyzes your sentiment using deep learning models, and provides personalized advice in your native language.

> **Status:** v1.2 (Dockerized & Production Ready) 🐳

---

## 🇬🇧 English Documentation

### 🚀 Key Features
* **🌍 Bi-Directional Translation:** Supports **Turkish, English, German, Spanish, and French**. Uses a "Pivot Translation" architecture (Source -> EN -> Target) to translate between any of these languages.
* **🧠 Sentiment & Mood Analysis:** Uses `nlptown/bert-base-multilingual-uncased-sentiment` to detect emotions with high accuracy.
* **💡 Personalized Advice:** Generates context-aware advice based on detected mood and translates it back to the user's language.
* **🐳 MLOps & Engineering:** Fully dockerized application with a modular structure, ready for deployment.
* **💻 Dual Interface:** Offers both a **Web UI (Gradio)** and a professional **CLI (Command Line Interface)**.

### 🛠️ Installation

#### Option A: Using Docker (Recommended)
Build and run the container to ensure consistency across environments.

```bash
# 1. Build Image
docker build -t dailymoodai:v1 .

# 2. Run Container
docker run -d -p 7860:7860 --name daily_mood_container dailymoodai:v1
```
Access the app at: http://127.0.0.1:7860


Option B: Local Setup
```bash
# 1. Install Dependencies
pip install -r requirements.txt
 ``` 
 ```bash
# 2. Run Application
python -m scripts.main ui
```


🖥️ CLI Usage
You can also use the tool directly from the terminal for batch processing or testing.
``` bash

# Get mood suggestion for text
python -m scripts.main suggest --text "Bugün harika hissediyorum" --lang tr

# Evaluate translation models (BLEU/ROUGE scores)
python -m scripts.main translate-eval --csv data/translation_eval.csv

# Check API cost summary (It's Free!)
python -m scripts.main cost-summary

```

🏗️ Tech Stack
Core: Python 3.10, PyTorch

Models: Hugging Face Transformers (MarianMT for translation, BERT for sentiment)

Interface: Gradio (Web), Argparse (CLI)

DevOps: Docker, Git
---

## 📊 Example Outputs / Örnek Çıktılar

### 1. Gradio Web Interface (Arayüz)
![Gradio UI](reports/demo_view.png)

### 2. Sentiment Analysis Performance (Başarı Matrisi)
This confusion matrix shows how accurately the model predicts emotions (Negative, Neutral, Positive).
*(Bu matris, modelin duyguları ne kadar doğru tahmin ettiğini gösterir.)*

![Confusion Matrix](reports/confusion_matrix.png)

### 3. API Cost & Latency (Maliyet Raporu)
Since we use local models, the cost is **$0.00**! The chart below confirms zero API spend.
*(Yerel modeller kullandığımız için maliyet **$0.00**'dır. Aşağıdaki grafik sıfır harcamayı doğrular.)*

![Cost Plot](reports/cost_plot.png)

---

🇹🇷 Türkçe Dokümantasyon
🚀 Öne Çıkan Özellikler
🌍 Çift Yönlü Çeviri: Türkçe, İngilizce, Almanca, İspanyolca ve Fransızca dilleri arasında köprü kurar. Hangi dilde yazarsanız yazın, sizi anlar ve cevabı kendi dilinizde verir.

🧠 Duygu Analizi: BERT tabanlı çok dilli modeller kullanarak ruh halinizi analiz eder.

💡 Kişisel Tavsiyeler: Ruh halinize uygun motivasyon cümleleri sunar.

🐳 MLOps ve Mühendislik: Proje, Docker ile konteynerize edilmiştir. Her ortamda sorunsuz çalışır.

💻 Çift Arayüz: Hem tarayıcı üzerinden (Gradio) hem de terminalden (CLI) kullanılabilir.

🛠️ Kurulum
Seçenek A: Docker Kullanarak (Önerilen)
Uygulamayı izole bir ortamda, hata almadan çalıştırmak için:

```bash
# 1. İmajı Oluştur
docker build -t dailymoodai:v1 .

# 2. Konteyneri Başlat
docker run -d -p 7860:7860 --name daily_mood_container dailymoodai:v1
```
Uygulamaya git: http://127.0.0.1:7860

Seçenek B: Yerel Kurulum
```bash
# 1. Kütüphaneleri Yükle
pip install -r requirements.txt

# 2. Uygulamayı Başlat
python -m scripts.main ui
```
📂 Project Structure / Proje Yapısı
DailyMoodAI/
├── data/               # Data for evaluation and suggestions
├── reports/            # Generated metrics, logs, and screenshots
├── scripts/            # Source code (inference, main, logger)
├── Dockerfile          # Docker configuration
├── requirements.txt    # Python dependencies
└── README.md           # Documentation

👤 Author
Esmanur Erdiş 
