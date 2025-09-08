from __future__ import annotations

from typing import Dict, Tuple, List
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import json
import re

# Route/cost logger
from scripts.route_logger import log_call

# 1) ÇEVİRİ MarianMT 
_SUPPORTED = {
    "tr": "Helsinki-NLP/opus-mt-tr-en",
    "de": "Helsinki-NLP/opus-mt-de-en",
    "es": "Helsinki-NLP/opus-mt-es-en",
    "en": None,  # İngilizce gelirse model gerekmez
}

@lru_cache(maxsize=None)
def _get_translation_pipe(src_lang: str):
    """Kaynak dil için çeviri pipeline'ını hazırla ve cache et."""
    src = (src_lang or "").lower()
    if src == "en":
        return None
    from transformers import pipeline
    model_id = _SUPPORTED.get(src)
    if model_id is None:
        return None
    # CPU kullanımı: device=-1
    return pipeline("translation", model=model_id, device=-1)

def translate_to_en(text: str, src_lang: str) -> str:
    """Metni İngilizceye çevir. Desteklenmeyen dil gelirse olduğu gibi döndürür."""
    if not text:
        return ""
    pipe = _get_translation_pipe(src_lang)
    if pipe is None:
        # İngilizce veya desteklenmeyen dil → çevirmeden döndür
        return text

    # Süre ölç + çeviri
    start = perf_counter()
    out = pipe(text, max_length=256)
    latency = perf_counter() - start

    hyp = out[0]["translation_text"]

    # Kaba token tahmini: boşlukla ayrılmış kelime sayısı
    tokens = len((text or "").split())

    # Lokal model → cost=0.0
    log_call(model=f"translation-{(src_lang or '').lower()}-en", tokens=tokens, cost=0.0, latency=latency)
    return hyp

# EN mood -> TR karşılık
_EN2TR = {
    "sad": "üzgün",
    "anxious": "kaygılı",
    "happy": "mutlu",
    "tired": "yorgun",
}

# Basit İngilizce anahtar kelime listeleri
_KW = {
    "tired":   {"tired", "sleepy", "exhausted", "fatigued", "weary", "drained"},
    "anxious": {"anxious", "nervous", "worried", "stressed", "panic", "tense"},
    "sad":     {"sad", "unhappy", "upset", "depressed", "down", "miserable"},
    "happy":   {"happy", "good", "great", "glad", "joy", "love", "excited", "awesome", "fine", "okay", "ok"},
}

def _normalize_en(s: str) -> List[str]:
    s = (s or "").lower()
    s = re.sub(r"[^a-z\s]", " ", s)  # basit temizlik
    return s.split()

def _rule_based_mood_en(text_en: str) -> str:
    """Çok basit, anlaşılır kural seti (junior seviye)."""
    toks = set(_normalize_en(text_en))

    # 1) Fiziksel yorgunluk işaretleri varsa önce onu seç
    if toks & _KW["tired"]:
        return "tired"
    # 2) Kaygı işaretleri
    if toks & _KW["anxious"]:
        return "anxious"

    # 3) Mutlu vs üzgün sayımı
    happy_hits = len(toks & _KW["happy"])
    sad_hits   = len(toks & _KW["sad"])
    if happy_hits > sad_hits:
        return "happy"
    if sad_hits > happy_hits:
        return "sad"

    # 4) Nötr: pozitif varsayım
    return "happy"

@lru_cache(maxsize=None)
def _load_suggestions(path: str = "data/suggestions.json") -> List[Dict]:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def suggest_mood_and_advice(user_text: str, lang: str = "tr") -> Tuple[str, str, str]:
    """
    Girdi: user_text + dili
    Çıkış: (mood_tr, suggestion_tr, translated_en)
    """
    # 1) İngilizce normalize et
    text_en = user_text if (lang or "").lower() == "en" else translate_to_en(user_text, lang)

    # 2) Mood tespit (EN)
    mood_en = _rule_based_mood_en(text_en)

    # 3) TR karşılık + öneriyi getir
    mood_tr = _EN2TR[mood_en]
    suggestions = _load_suggestions()
    suggestion_tr = next(
        (row["suggestion"] for row in suggestions if row.get("mood") == mood_tr),
        "Kendine nazik ol; kısa bir mola ver. 😊"
    )
    return mood_tr, suggestion_tr, text_en

# 3) SENTIMENT (gerçek model – çok dilli)
# Model: nlptown/bert-base-multilingual-uncased-sentiment
# Çıktıyı 1–5 yıldızdan pos/neg/neu'ya çeviriyoruz.
@lru_cache(maxsize=1)
def _get_sentiment_pipe():
    from transformers import pipeline
    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_id, device=-1)

def predict_sentiment(text: str, lang: str = "tr") -> Dict[str, float]:
    """
    Dönen sözlük:
      {"label": "pos|neg|neu", "proba": 0.0-1.0}
    """
    if not text:
        return {"label": "neu", "proba": 0.0}

    pipe = _get_sentiment_pipe()

    # Süre ölç + tahmin
    start = perf_counter()
    result = pipe(text[:512])[0] 
    latency = perf_counter() - start

    # Token tahmini
    tokens = len((text or "").split())

    # Logla (lokal model, cost=0.0)
    log_call(model="sentiment-nlptown-bert", tokens=tokens, cost=0.0, latency=latency)

    label = result["label"]
    score = float(result["score"])
    try:
        stars = int(label.split()[0])
    except Exception:
        return {"label": "neu", "proba": score}

    if stars in (4, 5):
        return {"label": "pos", "proba": score}
    if stars == 3:
        return {"label": "neu", "proba": score}
    return {"label": "neg", "proba": score}
