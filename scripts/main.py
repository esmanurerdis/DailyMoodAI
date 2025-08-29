# main.py
# DailyMoodAI - Basit öneri motoru (TR kısa metinler için char n-gram TF-IDF)
# Modlar: CLI ve Gradio UI
# Çalıştırma:
#   python main.py                 # CLI + ardından UI
#   python main.py --mode ui       # Sadece UI
#   python main.py --mode cli      # Sadece CLI

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import gradio as gr
except ImportError:
    gr = None  # UI gerekmiyorsa sorun değil; sadece CLI çalışır.

# -------------------------
# Yapılandırmalar
# -------------------------
DATA_PATH = Path("data/suggestions.json")  # Öneri dosyan burada dursun.


# -------------------------
# Yardımcı Fonksiyonlar
# -------------------------
def load_suggestions(path: Path) -> List[Dict[str, Any]]:
    """
    suggestions.json formatı:
    [
      {"mood": "üzgün",  "suggestion": "Derin bir nefes al..."},
      {"mood": "kaygılı", "suggestion": "5 dakikalık nefes egzersizi..."},
      ...
    ]
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Suggestions file not found: {path}\n"
            f"Lütfen {path} dosyasını oluştur: örnek içerik ->\n"
            f'[{{"mood":"üzgün","suggestion":"Derin bir nefes al..."}}]'
        )
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Basit şema kontrolü
    if not isinstance(data, list) or not all(isinstance(x, dict) for x in data):
        raise ValueError("Invalid suggestions.json format: root list of objects required.")
    for i, item in enumerate(data):
        if "mood" not in item or "suggestion" not in item:
            raise ValueError(f"suggestions[{i}] missing 'mood' or 'suggestion' keys.")
    return data


def build_vectorizer() -> TfidfVectorizer:
    """
    Türkçe ve kısa metinlerde daha iyi eşleşme için karakter n-gram TF-IDF.
    """
    return TfidfVectorizer(analyzer="char", ngram_range=(3, 5))


def best_suggestion(user_text: str, suggestions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Kullanıcı metnini en yakın ruh hali ile eşleştirir.
    Dönen sözlük: { matched_mood, suggestion, score }
    """
    user_text = (user_text or "").strip().lower()
    if not user_text:
        return {
            "matched_mood": "",
            "suggestion": "Lütfen nasıl hissettiğini bir-iki kelimeyle yaz.",
            "score": 0.0,
        }

    moods = [s["mood"] for s in suggestions]
    vect = build_vectorizer()
    X = vect.fit_transform(moods + [user_text])
    sim = cosine_similarity(X[-1], X[:-1]).ravel()
    idx = int(sim.argmax())
    return {
        "matched_mood": moods[idx],
        "suggestion": suggestions[idx]["suggestion"],
        "score": float(sim[idx]),
    }


# -------------------------
# CLI ve UI
# -------------------------
def run_cli(suggestions: List[Dict[str, Any]]) -> None:
    try:
        user_input = input("Bugün nasıl hissediyorsun? ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nİptal edildi.")
        return

    result = best_suggestion(user_input, suggestions)
    print(f"Eşleşen ruh hali: {result['matched_mood']} (skor: {result['score']:.2f})")
    print("Tavsiyen:", result["suggestion"])


def run_ui(suggestions: List[Dict[str, Any]]) -> None:
    if gr is None:
        print("Gradio yüklü değil. UI için: pip install gradio")
        return

    def ui_predict(txt: str) -> str:
        r = best_suggestion(txt, suggestions)
        return (
            f"Eşleşen ruh hali: {r['matched_mood']} (skor: {r['score']:.2f})\n"
            f"Tavsiye: {r['suggestion']}"
        )

    gr.Interface(
        fn=ui_predict,
        inputs=gr.Textbox(lines=2, label="Bugün nasıl hissediyorsun?"),
        outputs=gr.Textbox(lines=4, label="Öneri"),
        title="DailyMoodAI",
        description="Kısa bir duygu/mood yaz; en yakın ruh hali ve öneriyi verelim.",
        allow_flagging="never",
    ).launch()


# -------------------------
# Entry Point
# -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="DailyMoodAI - CLI & Web UI")
    p.add_argument(
        "--mode",
        choices=["cli", "ui", "both"],
        default="both",
        help="Çalıştırma modu: cli | ui | both (varsayılan: both)",
    )
    p.add_argument(
        "--data",
        type=str,
        default=str(DATA_PATH),
        help="Öneri JSON dosyasının yolu (varsayılan: data/suggestions.json)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    suggestions = load_suggestions(Path(args.data))

    if args.mode in ("cli", "both"):
        run_cli(suggestions)
    if args.mode in ("ui", "both"):
        run_ui(suggestions)


if __name__ == "__main__":
    main()
