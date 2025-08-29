import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = Path("data/suggestions.json")

def load_suggestions(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Suggestions file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def best_suggestion(user_text: str, suggestions: list[dict]) -> dict:
    moods = [s["mood"] for s in suggestions]
    # char n-gram, kısa/TR metinlerde daha sağlam eşleşir
    vect = TfidfVectorizer(analyzer="char", ngram_range=(3,5))
    X = vect.fit_transform(moods + [user_text])
    sim = cosine_similarity(X[-1], X[:-1]).ravel()
    idx = int(sim.argmax())
    return suggestions[idx] | {"matched_mood": moods[idx], "score": float(sim[idx])}

if __name__ == "__main__":
    user_input = input("Bugün nasıl hissediyorsun? ").strip().lower()
    suggestions = load_suggestions(DATA_PATH)
    result = best_suggestion(user_input, suggestions)
    print(f"Eşleşen ruh hali: {result['matched_mood']} (skor: {result['score']:.2f})")
    print("Tavsiyen:", result["suggestion"])

