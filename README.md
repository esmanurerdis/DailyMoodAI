#🧠 DailyMoodAI – Mood Suggestion Chatbot

DailyMoodAI, kullanıcının ruh halini kısa bir metinle ifade etmesine olanak tanır ve en uygun öneriyi döndürür.
Model, TF-IDF + Cosine Similarity yaklaşımıyla ruh hallerini eşleştirir.
Ek olarak, Gradio arayüzü sayesinde hem terminal hem de web tabanlı kullanım mümkündür.

🚀 Özellikler

CLI & Web Arayüzü: Terminalden veya tarayıcıdan çalıştırılabilir.

Türkçe Mood Önerileri: Üzgün, kaygılı, mutlu, yorgun gibi ruh hallerini yakalar.

TF-IDF + Cosine Similarity: Kısa metinlerde doğru eşleşme için char-ngram tabanlı vektörleştirme.

Gradio UI: Kullanıcı dostu web arayüzü.

Kolay Genişletilebilirlik: Yeni mood ve öneriler JSON dosyasına eklenerek hızlıca çoğaltılabilir.

📂 Proje Yapısı
DailyMoodAI/
├── main.py               # Ana uygulama (CLI + Gradio UI)
├── data/
│   └── suggestions.json  # Mood & öneriler
├── requirements.txt      # Bağımlılıklar
└── README.md             # Bu dosya

🔧 Kurulum ve Çalıştırma
1. Bağımlılıkları yükle
pip install -r requirements.txt

2. Çalıştır
# CLI ve UI birlikte
python main.py

# Sadece web arayüzü
python main.py --mode ui

# Sadece terminal modu
python main.py --mode cli

🖼 Demo

👉 Buraya bir screenshot veya GIF ekle:

Terminal ekran görüntüsü

Gradio arayüzünden bir örnek

📊 Kullanılan Teknolojiler

Python 3.9+

scikit-learn (TF-IDF, cosine similarity)

Gradio (web UI)

💡 Örnek Kullanım
Bugün nasıl hissediyorsun? kaygılı
Eşleşen ruh hali: kaygılı (skor: 0.88)
Tavsiyen: 5 dakikalık bir nefes egzersizi yap ya da dışarıda kısa bir yürüyüşe çık.

🏷 Lisans

MIT License
