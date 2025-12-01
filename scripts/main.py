"""
DailyMoodAI - Main CLI Application
Multi-lingual support with Target Language Selection.
"""

import argparse
import sys
from pathlib import Path
import json
import pandas as pd

# Imports
try:
    from scripts.inference import (
        translate_text,
        predict_sentiment,
        suggest_mood_and_advice,
        get_model_info,
        get_supported_languages,
    )
except ImportError:
    from inference import (
        translate_text,
        predict_sentiment,
        suggest_mood_and_advice,
        get_model_info,
        get_supported_languages,
    )

VERSION = "1.2.0"

# =============================================================================
# TRANSLATION EVALUATION
# =============================================================================

def translate_eval(csv_path: str, out_json: str = "reports/bleu_rouge.json"):
    """
    Evaluate translation quality using BLEU and ROUGE metrics.
    Test converts Source -> English to compare with Reference English.
    """
    import sacrebleu
    from rouge_score import rouge_scorer

    print(f"📊 Evaluating translations from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    required_cols = ["src_lang", "src_text", "ref_en"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: CSV must have columns: {required_cols}")
        return
    
    print("🔄 Translating texts to English for evaluation...")
    refs = df["ref_en"].tolist()
    
    hyps = [
        translate_text(t, source_lang=l, target_lang="en") 
        for t, l in zip(df["src_text"], df["src_lang"])
    ]
    
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], 
        use_stemmer=True
    )
    r1 = r2 = rL = 0.0
    for r, h in zip(refs, hyps):
        s = scorer.score(r, h)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rL += s["rougeL"].fmeasure
    
    n = max(len(refs), 1)
    metrics = {
        "BLEU": round(bleu, 2),
        "ROUGE1_F1": round(r1 / n, 4),
        "ROUGE2_F1": round(r2 / n, 4),
        "ROUGEL_F1": round(rL / n, 4),
        "num_samples": n
    }
    
    print(f"\n✅ Results: BLEU Score: {metrics['BLEU']:.2f}")


# =============================================================================
# SENTIMENT EVALUATION
# =============================================================================

def sentiment_eval(
    csv_path: str,
    out_json: str = "reports/sentiment_report.json",
    out_cm: str = "reports/confusion_matrix.png"
):
    """Evaluate sentiment analysis accuracy."""
    from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    print(f"📊 Evaluating sentiment from: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"❌ Error: File not found: {csv_path}")
        return
    
    required_cols = ["text", "lang", "true_label"]
    if not all(col in df.columns for col in required_cols):
        print(f"❌ Error: CSV must have columns: {required_cols}")
        return
    
    print("🔄 Predicting sentiments...")
    y_true = df["true_label"].astype(str).tolist()
    y_pred = [
        predict_sentiment(t, l)["label"] 
        for t, l in zip(df["text"], df["lang"])
    ]
    
    report = classification_report(
        y_true, y_pred, 
        output_dict=True, 
        zero_division=0
    )
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    # Save Confusion matrix
    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    
    plt.figure(figsize=(8, 6))
    disp.plot(values_format="d", cmap="Blues")
    plt.title("Sentiment Analysis - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_cm, dpi=150)
    plt.close()
    
    accuracy = report.get("accuracy", 0)
    print(f"\n✅ Results: Accuracy: {accuracy:.2%}")


# =============================================================================
# COST SUMMARY
# =============================================================================

def summarize_cost(
    log_path="reports/route_log.csv",
    out_json="reports/cost_summary.json",
    out_png="reports/cost_plot.png"
):
    """Generate cost summary."""
    import matplotlib.pyplot as plt

    p = Path(log_path)
    
    if not p.exists():
        print(f"⚠️  Warning: {log_path} not found. No API calls logged yet.")
        return
    
    print(f"📊 Analyzing costs from: {log_path}")
    df = pd.read_csv(p)
    
    summary = {
        "total_calls": int(len(df)),
        "total_cost": float(df["cost"].sum()),
        "avg_latency": round(float(df["latency"].mean()), 3)
    }
    
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Simple bar chart (if data is available)
    try:
        per_model = df.groupby("model")[["cost"]].sum().reset_index()
        fig, ax = plt.subplots(figsize=(10, 6))
        per_model.plot(x="model", y="cost", kind="bar", ax=ax, legend=False)
        plt.savefig(out_png, dpi=150)
        plt.close()
    except:
        pass # Ignore plot errors if data is trivial

    print(f"\n✅ Cost Summary: Total Cost: ${summary['total_cost']:.2f} (Avg Latency: {summary['avg_latency']:.3f}s)")

def serve_ui(port: int = 7860, share: bool = False):
    try:
        import gradio as gr
    except ImportError:
        print("❌ Error: Gradio not installed.")
        return
    
    print(f"🚀 Starting DailyMoodAI on port {port}...")
    
    def _predict(text, input_lang, target_lang):
        text = (text or "").strip()
        if not text:
            return "—", "—", "Empty input"
        
        try:
            # GÜNCELLENDİ: Artık hedef dili de gönderiyoruz
            mood_en, advice_local, trans_text = suggest_mood_and_advice(
                user_text=text, 
                input_lang=input_lang, 
                target_lang=target_lang
            )
            return mood_en, advice_local, trans_text
        except Exception as e:
            return "Error", f"Failed: {e}", ""
    
    langs = get_supported_languages() # ['tr', 'de', 'es', 'fr', 'en']
    
    demo = gr.Interface(
        fn=_predict,
        inputs=[
            gr.Textbox(label="How do you feel?", placeholder="Bugün nasılsın? / How are you?", lines=3),
            gr.Dropdown(choices=langs, label="Input Language ", value="tr"),
            gr.Dropdown(choices=langs, label="Translate To ", value="en") # YENİ ALAN
        ],
        outputs=[
            gr.Label(label="Detected Mood "),
            gr.Textbox(label="My Advice 💡", lines=2),
            gr.Textbox(label="Translation Output 🌍", lines=2)
        ],
        title="🌟 DailyMoodAI ",
        description=" ",
        examples=[
            ["Bugün harika hissediyorum!", "tr", "de"], # Türkçe gir, Almanca çeviri al
            ["I am very tired today.", "en", "es"],     # İngilizce gir, İspanyolca çeviri al
            ["Ich bin ein Berliner.", "de", "tr"],      # Almanca gir, Türkçe çeviri al
        ],
        theme=gr.themes.Soft(),
    )
    
    demo.launch(server_name="0.0.0.0", server_port=port, share=share)

# ... (Rest of CLI setup code remains similar) ...

def build_parser():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    # translate-eval komutunu ekle
    t = sub.add_parser(
        "translate-eval",
        help="Evaluate translation quality (BLEU/ROUGE)"
    )
    t.add_argument(
        "--csv",
        default="data/translation_eval.csv",
        help="CSV with columns: src_lang, src_text, ref_en"
    )
    t.set_defaults(func=lambda a: translate_eval(a.csv))
    
    # sentiment-eval komutunu ekle
    s = sub.add_parser(
        "sentiment-eval",
        help="Evaluate sentiment analysis accuracy"
    )
    s.add_argument(
        "--csv",
        default="data/sentiment_eval.csv",
        help="CSV with columns: text, lang, true_label"
    )
    s.set_defaults(func=lambda a: sentiment_eval(a.csv))
    
    # cost-summary komutunu ekle
    c = sub.add_parser(
        "cost-summary",
        help="Show API usage summary (spoiler: it's $0!)"
    )
    c.set_defaults(func=lambda a: summarize_cost())
    
    # info komutunu ekle (Opsiyonel)
    i = sub.add_parser("info", help="Show model information")
    i.set_defaults(func=lambda a: _cmd_info())
    
    u = sub.add_parser("ui", help="Launch Web UI")
    u.add_argument("--port", type=int, default=7860)
    u.add_argument("--share", action="store_true")
    u.set_defaults(func=lambda a: serve_ui(a.port, a.share))
    
    return p

def main():
    try:
        args = build_parser().parse_args()
        args.func(args)
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    main()