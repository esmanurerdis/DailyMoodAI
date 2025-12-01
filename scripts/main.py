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

# ... (Evaluation functions can stay same or be minimized for now) ...

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