import argparse
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import sacrebleu
from rouge_score import rouge_scorer

# Proje içi importlar
from scripts.inference import (
    translate_to_en,
    predict_sentiment,
    suggest_mood_and_advice,
)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# 1) ÇEVİRİ DEĞERLENDİRME (BLEU/ROUGE)
def translate_eval(csv_path: str, out_json: str = "reports/bleu_rouge.json"):
    """
    CSV kolonları: src_lang, src_text, ref_en
    """
    df = pd.read_csv(csv_path)

    refs = df["ref_en"].tolist()
    hyps = [translate_to_en(t, l) for t, l in zip(df["src_text"], df["src_lang"])]

    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score

    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    r1 = r2 = rL = 0.0
    for r, h in zip(refs, hyps):
        s = scorer.score(r, h)
        r1 += s["rouge1"].fmeasure
        r2 += s["rouge2"].fmeasure
        rL += s["rougeL"].fmeasure
    n = max(len(refs), 1)
    metrics = {
        "BLEU": bleu,
        "ROUGE1_F1": r1 / n,
        "ROUGE2_F1": r2 / n,
        "ROUGEL_F1": rL / n,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    pd.DataFrame({
        "src_lang": df["src_lang"],
        "src_text": df["src_text"],
        "ref_en": refs,
        "hyp_en": hyps
    }).to_csv("reports/bleu_rouge_breakdown.csv", index=False)

    print(f"[OK] Kaydedildi: {out_json} ve reports/bleu_rouge_breakdown.csv")


# 2) SENTIMENT DEĞERLENDİRME + CONFUSION MATRIX
def sentiment_eval(csv_path: str,
                   out_json: str = "reports/sentiment_report.json",
                   out_cm: str = "reports/confusion_matrix.png"):
    """
    CSV kolonları: text, lang, true_label
    """
    df = pd.read_csv(csv_path)
    y_true = df["true_label"].astype(str).tolist()
    y_pred = [predict_sentiment(t, l)["label"] for t, l in zip(df["text"], df["lang"])]

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    labels = sorted(list(set(y_true) | set(y_pred)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    plt.figure()
    disp.plot(values_format="d")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(out_cm)
    plt.close()

    print(f"[OK] Kaydedildi: {out_json} ve {out_cm}")


# 3) GRADIO ARAYÜZÜ
def serve_ui(port: int = 7860):
    import gradio as gr

    def _predict(text, lang):
        text = (text or "").strip()
        if not text:
            return "—", "—", "Boş giriş"
        mood_tr, suggestion_tr, text_en = suggest_mood_and_advice(text, lang)
        return mood_tr, suggestion_tr, text_en

    demo = gr.Interface(
        fn=_predict,
        inputs=[gr.Textbox(label="Text"),
                gr.Dropdown(["tr","en","de","es"], label="Language", value="tr")],
        outputs=[gr.Label(label="Mood (TR)"),
                 gr.Textbox(label="Suggestion (TR)", lines=2),
                 gr.Textbox(label="Translated to English", lines=2)],
        title="DailyMoodAI — Translate & Suggest (Junior)"
    )
    demo.launch(server_name="0.0.0.0", server_port=port, share=False)


# 4) ROUTE/COST ÖZETİ (opsiyonel)
def summarize_cost(log_path="reports/route_log.csv",
                   out_json="reports/cost_summary.json",
                   out_png="reports/cost_plot.png"):
    """
    'reports/route_log.csv' dosyasından maliyet/çağrı özetini üretir.
    CSV beklenen kolonlar: timestamp, model, tokens, cost, latency
    """
    p = Path(log_path)
    if not p.exists():
        print(f"[WARN] {log_path} bulunamadı. Önce log üretmelisin (log_call).")
        return

    df = pd.read_csv(p)

    summary = {
        "total_calls": int(len(df)),
        "total_tokens": int(df["tokens"].sum()),
        "total_cost": float(df["cost"].sum()),
        "avg_latency": float(df["latency"].mean())
    }

    per_model = df.groupby("model")[["tokens", "cost"]].sum().reset_index()
    summary["per_model"] = per_model.to_dict(orient="records")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # grafik: model bazlı toplam cost
    ax = per_model.plot(x="model", y="cost", kind="bar", legend=False)
    ax.set_ylabel("Total Cost (USD)")
    ax.set_title("Route Cost by Model")
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()

    print(f"[OK] Kaydedildi: {out_json} ve {out_png}")


# 5) CLI Kurulumu
def build_parser():
    p = argparse.ArgumentParser(description="DailyMoodAI — junior seviye CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    # translate-eval
    t = sub.add_parser("translate-eval", help="BLEU/ROUGE hesapla")
    t.add_argument("--csv", default="data/translation_eval.csv",
                   help="Kolonlar: src_lang, src_text, ref_en")
    t.set_defaults(func=lambda a: translate_eval(a.csv))

    # sentiment-eval
    s = sub.add_parser("sentiment-eval", help="Classification report + confusion matrix")
    s.add_argument("--csv", default="data/sentiment_eval.csv",
                   help="Kolonlar: text, lang, true_label")
    s.set_defaults(func=lambda a: sentiment_eval(a.csv))

    # ui
    u = sub.add_parser("ui", help="Gradio arayüzü")
    u.add_argument("--port", type=int, default=7860)
    u.set_defaults(func=lambda a: serve_ui(a.port))

    # suggest
    sg = sub.add_parser("suggest", help="Metni işle ve öneri üret")
    sg.add_argument("--text", required=True)
    sg.add_argument("--lang", default="tr")
    sg.set_defaults(func=lambda a: print(suggest_mood_and_advice(a.text, a.lang)))

    # cost-summary
    c = sub.add_parser("cost-summary", help="RouteLLM çağrı maliyet özeti üret")
    c.add_argument("--log", default="reports/route_log.csv")
    c.add_argument("--json", default="reports/cost_summary.json")
    c.add_argument("--png", default="reports/cost_plot.png")
    c.set_defaults(func=lambda a: summarize_cost(a.log, a.json, a.png))

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
