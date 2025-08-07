import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Kullanıcıdan ruh hali al
user_input = input("Bugün nasıl hissediyorsun? ")

# JSON dosyasından önerileri al
with open("data/suggestions.json", "r", encoding="utf-8") as f:
    suggestions = json.load(f)

moods = [s["mood"] for s in suggestions]
texts = moods + [user_input]

# TF-IDF ve cosine similarity
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(texts)
similarity = cosine_similarity(vectors[-1], vectors[:-1])

# En benzer mood'u bul
best_match = similarity.argmax()
print("Tavsiyen:", suggestions[best_match]["suggestion"])
