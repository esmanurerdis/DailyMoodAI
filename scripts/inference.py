"""
DailyMoodAI - Core Inference Module
Supports bi-directional and cross-lingual translation via English pivot.
"""

from __future__ import annotations

from typing import Dict, Tuple, List, Optional
from functools import lru_cache
from pathlib import Path
from time import perf_counter
import json
import re
import logging
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Dummy logger setup
try:
    from scripts.route_logger import log_call
except ImportError:
    try:
        from route_logger import log_call
    except ImportError:
        def log_call(**kwargs): pass

# =============================================================================
# 1. TRANSLATION LOGIC (Smart Pivot)
# =============================================================================

_LANG_MODELS = {
    "tr": {"to_en": "Helsinki-NLP/opus-mt-tr-en", "from_en": "Helsinki-NLP/opus-mt-en-tr"},
    "de": {"to_en": "Helsinki-NLP/opus-mt-de-en", "from_en": "Helsinki-NLP/opus-mt-en-de"},
    "es": {"to_en": "Helsinki-NLP/opus-mt-es-en", "from_en": "Helsinki-NLP/opus-mt-en-es"},
    "fr": {"to_en": "Helsinki-NLP/opus-mt-fr-en", "from_en": "Helsinki-NLP/opus-mt-en-fr"},
    "en": None
}

@lru_cache(maxsize=None)
def _get_translation_pipe(model_id: str):
    if not model_id: return None
    from transformers import pipeline
    logger.info(f"Loading translation model: {model_id}")
    return pipeline("translation", model=model_id, device=-1)

def translate_text(text: str, source_lang: str, target_lang: str) -> str:
    """
    Translates text from Source -> Target.
    If direct translation isn't possible, uses English as a pivot (Src -> EN -> Tgt).
    """
    if not text or not text.strip(): return ""
    
    src = (source_lang or "").lower()
    tgt = (target_lang or "").lower()
    
    # 1. Same language? Return as is.
    if src == tgt:
        return text
    
    # 2. Cross-Lingual Case (e.g., TR -> DE)
    # Neither side is English, so we must pivot: TR -> EN -> DE
    if src != "en" and tgt != "en":
        # Step A: Src -> EN
        intermediate_text = translate_text(text, src, "en")
        # Step B: EN -> Tgt
        final_text = translate_text(intermediate_text, "en", tgt)
        return final_text

    # 3. Direct Translation (X -> EN or EN -> X)
    model_id = None
    if tgt == "en":
        lang_config = _LANG_MODELS.get(src)
        if lang_config: model_id = lang_config.get("to_en")
    elif src == "en":
        lang_config = _LANG_MODELS.get(tgt)
        if lang_config: model_id = lang_config.get("from_en")
            
    if not model_id:
        logger.warning(f"No model found for {src} -> {tgt}")
        return text

    try:
        pipe = _get_translation_pipe(model_id)
        start = perf_counter()
        # Run translation
        out = pipe(text[:512], max_length=512)
        latency = perf_counter() - start
        
        log_call(model=model_id, tokens=len(text.split()), cost=0.0, latency=latency)
        return out[0]["translation_text"]
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

# =============================================================================
# 2. SENTIMENT ANALYSIS
# =============================================================================

@lru_cache(maxsize=1)
def _get_sentiment_pipe():
    from transformers import pipeline
    model_id = "nlptown/bert-base-multilingual-uncased-sentiment"
    return pipeline("sentiment-analysis", model=model_id, device=-1)

def predict_sentiment(text: str, lang: str = "tr") -> Dict[str, float]:
    if not text or not text.strip(): return {"label": "neu", "proba": 0.0}
    try:
        pipe = _get_sentiment_pipe()
        result = pipe(text[:512])[0]
        label = result["label"]
        score = float(result["score"])
        try:
            stars = int(label.split()[0])
        except:
            return {"label": "neu", "proba": score}
        
        if stars in (4, 5): return {"label": "pos", "proba": score}
        elif stars == 3: return {"label": "neu", "proba": score}
        else: return {"label": "neg", "proba": score}
    except Exception as e:
        logger.error(f"Sentiment failed: {e}")
        return {"label": "neu", "proba": 0.0}

# =============================================================================
# 3. MOOD & ADVICE LOGIC
# =============================================================================

_MOOD_KEYWORDS = {
    "tired": {"tired", "sleepy", "exhausted", "fatigued", "weary"},
    "anxious": {"anxious", "nervous", "worried", "stressed", "panic"},
    "sad": {"sad", "unhappy", "upset", "depressed", "down", "blue"},
    "happy": {"happy", "good", "great", "glad", "joy", "excited"}
}

def _detect_mood_rule_based(text_en: str) -> str:
    tokens = set(re.sub(r"[^a-z\s]", " ", text_en.lower()).split())
    if tokens & _MOOD_KEYWORDS["tired"]: return "tired"
    if tokens & _MOOD_KEYWORDS["anxious"]: return "anxious"
    if len(tokens & _MOOD_KEYWORDS["sad"]) > len(tokens & _MOOD_KEYWORDS["happy"]): return "sad"
    return "happy"

@lru_cache(maxsize=None)
def _load_suggestions(path: str = "data/suggestions.json") -> Dict:
    p = Path(path)
    if not p.exists(): p = Path(__file__).parent.parent / "data" / "suggestions.json"
    if not p.exists(): return {}
    with p.open("r", encoding="utf-8") as f: return json.load(f)

def suggest_mood_and_advice(user_text: str, input_lang: str = "tr", target_lang: str = "en") -> Tuple[str, str, str]:
    """
    1. Translate Input -> EN (for analysis)
    2. Analyze Mood
    3. Generate Advice (in EN)
    4. Translate Advice -> Input Lang (for user)
    5. Translate Input -> Target Lang (for user requested translation)
    """
    try:
        # 1. Analysis is always done in English
        text_for_analysis = translate_text(user_text, input_lang, "en")
        
        # 2. Detect Mood
        mood_en = _detect_mood_rule_based(text_for_analysis)
        
        # 3. Get Advice
        mood_map = {"happy": "positive", "sad": "negative", "anxious": "negative", "tired": "neutral"}
        category = mood_map.get(mood_en, "neutral")
        suggestions = _load_suggestions()
        options = suggestions.get(category, ["Stay positive!"])
        advice_en = random.choice(options)
        
        # 4. Translate Advice back to User's Native Language
        advice_local = translate_text(advice_en, "en", input_lang)
        
        # 5. Translate Original Input to Requested Target Language
        translated_input = translate_text(user_text, input_lang, target_lang)
        
        return mood_en, advice_local, translated_input
        
    except Exception as e:
        logger.error(f"Logic error: {e}")
        return "error", "Error.", user_text

# =============================================================================
# 4. UTILS
# =============================================================================

def get_supported_languages() -> List[str]:
    return list(_LANG_MODELS.keys())

def get_model_info() -> Dict[str, str]:
    return {"status": "Active", "mode": "Bi-Directional Translation"}