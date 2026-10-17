"""Cinemood (سینمود): face -> mood -> movie recommendations. Run: python app.py"""
import html
import json
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "cinemood.keras"
DATASET_PATH = BASE_DIR / "movies_sample.json"

MOODS = ["angry", "happy", "neutral", "sad", "surprise"]
MOOD_FA = {"happy": "شاد", "sad": "غمگین", "angry": "عصبانی", "surprise": "متعجب", "neutral": "خنثی"}
MOOD_COLOR = {"happy": "#f59e0b", "sad": "#3b82f6", "angry": "#ef4444", "surprise": "#8b5cf6", "neutral": "#64748b"}
GENRE_FA = {
    "animation": "انیمیشن", "comedy": "کمدی", "drama": "درام", "adventure": "ماجراجویی",
    "sci-fi": "علمی‌تخیلی", "action": "اکشن", "music": "موسیقی", "family": "خانوادگی",
    "fantasy": "فانتزی", "mystery": "معمایی", "romance": "عاشقانه", "thriller": "هیجان‌انگیز",
}
ALL_GENRES = "همه ژانرها"

for path in (MODEL_PATH, DATASET_PATH):
    if not path.exists():
        raise FileNotFoundError(f"فایل پیدا نشد: {path} (مسیر فایل را بررسی کن)")

MODEL = tf.keras.models.load_model(MODEL_PATH)
with open(DATASET_PATH, encoding="utf-8") as f:
    MOVIES = json.load(f)
FACE_DETECTOR = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def prepare_face(image, crop):
    """PIL image -> model input (1, 48, 48, 1), matching FER-2013 preprocessing."""
    image = ImageOps.exif_transpose(image).convert("L")
    found = False
    if crop:
        faces = FACE_DETECTOR.detectMultiScale(
            np.asarray(image, dtype=np.uint8), 1.1, 5, minSize=(40, 40)
        )
        if len(faces):
            x, y, w, h = (int(v) for v in max(faces, key=lambda b: b[2] * b[3]))
            image, found = image.crop((x, y, x + w, y + h)), True
    array = np.asarray(image.resize((48, 48), Image.LANCZOS), dtype=np.float32) / 255.0
    return array[np.newaxis, :, :, np.newaxis], found


def pick_movies(mood, genre):
    movies = [m for m in MOVIES if mood in m["moods"]]
    note = ""
    if genre != ALL_GENRES:
        by_genre = [m for m in movies if genre in m["genres"]]
        if by_genre:
            movies = by_genre
        else:
            note = f"فیلمی با ژانر «{GENRE_FA.get(genre, genre)}» برای این حال پیدا نشد؛ همه‌ی فیلم‌های این حال نمایش داده شد."
    return sorted(movies, key=lambda m: m["rating"], reverse=True), note


def esc(value):
    return html.escape(str(value))


def chip(mood, extra=""):
    return f'<span class="chip {extra}" style="background:{MOOD_COLOR[mood]}">{MOOD_FA[mood]}</span>'


def render_mood(mood, confidence, face_note):
    low = '<div class="note">مدل زیاد مطمئن نیست؛ یک عکس واضح‌تر و رو‌به‌رو امتحان کن.</div>' if confidence < 0.5 else ""
    face = f'<div class="note">{esc(face_note)}</div>' if face_note else ""
    return f'''<div class="rtl mood-badge" style="--accent:{MOOD_COLOR[mood]}">
      <div class="small-text">حال تشخیص‌داده‌شده</div>
      <div class="mood-name" style="color:{MOOD_COLOR[mood]}">{MOOD_FA[mood]}</div>
      <div>اطمینان مدل: <b>{confidence:.0%}</b></div></div>{low}{face}'''


def render_scores(probs):
    rows = ""
    for i in np.argsort(probs)[::-1]:
        mood, pct = MOODS[i], float(probs[i]) * 100
        rows += (f'<div class="score-row">{chip(mood, "fixed")}<div class="bar">'
                 f'<i style="width:{pct:.1f}%;background:{MOOD_COLOR[mood]}"></i></div><b>{pct:.0f}٪</b></div>')
    return f'<div class="rtl card">{rows}</div>'


def render_results(mood, movies, note):
    if not movies:
        return '<div class="rtl card">در دیتاست فیلمی برای این حال ثبت نشده است.</div>'
    cards = ""
    for m in movies:
        genres = "، ".join(GENRE_FA.get(g, g) for g in m["genres"])
    cards = ""
    for m in movies:
        genres = "، ".join(GENRE_FA.get(g, g) for g in m["genres"])
        poster = str(m.get("poster") or "").strip()
        poster_html = (f'<img class="poster" src="{esc(poster)}" alt="" loading="lazy">'
                       if poster.startswith(("http://", "https://")) else '<div class="poster empty">🎬</div>')
        cards += f'''<article class="rtl card movie" style="--accent:{MOOD_COLOR[mood]}">{poster_html}<div>
          <div class="movie-title">{esc(m['title_fa'])}</div>
          <div class="small-text" dir="ltr">{esc(m['title'])} · {esc(m['year'])}</div>
          <div>⭐ <b>{esc(m['rating'])}</b></div><div class="small-text">{esc(genres)}</div>
          {chip(mood, "small")}</div></article>'''
    note_html = f'<div class="rtl note warn">{esc(note)}</div>' if note else ""
    return f'''<div class="rtl"><h2>فیلم‌های مناسب حال «{MOOD_FA[mood]}»</h2>
      <p class="small-text">{len(movies)} فیلم از movies_sample.json، مرتب‌شده بر اساس امتیاز</p></div>
      {note_html}<div class="movie-grid">{cards}</div>'''


def analyze(image, genre, crop):
    if image is None:
        raise gr.Error("لطفاً ابتدا یک تصویر از چهره انتخاب کن.")
    x, found = prepare_face(image, crop)
    probs = MODEL.predict(x, verbose=0)[0]
    if len(probs) != len(MOODS):
        raise gr.Error(f"مدل {len(probs)} خروجی دارد ولی {len(MOODS)} حال تعریف شده است.")
    best = int(np.argmax(probs))
    mood, confidence = MOODS[best], float(probs[best])
    result = {"mood": mood, "confidence": round(confidence, 2)}
    movies, note = pick_movies(mood, genre)
    face_note = ""
    if crop:
        face_note = "چهره پیدا و برش داده شد ✓" if found else "چهره‌ای پیدا نشد؛ کل تصویر تحلیل شد."
    return result, render_mood(mood, confidence, face_note), render_scores(probs), render_results(mood, movies, note)


def analyze_live(frame, state, genre):
    """Analyze webcam frames and smooth the last eight predictions."""
    if frame is None:
        return state, gr.update(), gr.update(), gr.update(), gr.update()
    # Gradio webcam frames are usually RGB numpy arrays.
    x, found = prepare_face(Image.fromarray(frame), True)
    if not found:
        state["hist"] = []
        return state, gr.update(), '<div class="rtl note">چهره‌ای پیدا نشد؛ مقابل دوربین بنشین.</div>', gr.update(), gr.update()
    state["hist"] = (state["hist"] + [MODEL(x, training=False).numpy()[0]])[-8:]
    probs = np.mean(state["hist"], axis=0)
    best = int(np.argmax(probs))
    mood, confidence = MOODS[best], float(probs[best])
    results = gr.update()
    if mood != state["mood"]:
        movies, note = pick_movies(mood, genre)
        results = render_results(mood, movies, note)
        state["mood"] = mood
    return state, {"mood": mood, "confidence": round(confidence, 2)}, render_mood(mood, confidence, "تحلیل زنده ✓"), render_scores(probs), results


LOGO = '''<svg viewBox="0 0 120 120" xmlns="http://www.w3.org/2000/svg"><rect x="2" y="2" width="116" height="116" rx="28" fill="#ff7a59"/><g fill="#2a2f4f">{}</g><circle cx="48" cy="50" r="6.5" fill="#fff"/><circle cx="72" cy="50" r="6.5" fill="#fff"/><path d="M40 68 Q60 90 80 68" stroke="#fff" stroke-width="8" stroke-linecap="round" fill="none"/></svg>'''.format(
    "".join(f'<rect x="{x}" y="{y}" width="10" height="10" rx="3"/>' for x in (12, 98) for y in (20, 40, 60, 80))
)
HERO = f'''<div class="strip"></div><section class="rtl hero"><div class="hero-text">
<h1>سینمود</h1><h2>حالت رو از چهره‌ت می‌فهمه؛ فیلم مناسبش رو پیشنهاد می‌ده</h2>
<p>عکس چهره‌ات رو بارگذاری کن یا از وب‌کم استفاده کن.</p><div class="small-text">پنج حالی که سینمود می‌شناسه:</div>
<div class="chips">{"".join(chip(m) for m in ["happy", "sad", "angry", "surprise", "neutral"])}</div>
</div><div class="logo">{LOGO}</div></section>'''
CSS = r'''
@import url('https://fonts.googleapis.com/css2?family=Lalezar:wght@400;600;800&display=swap');
:root, .dark, .gradio-container { --font: 'Lalezar', 'Tahoma', sans-serif; --body-background-fill: transparent; --block-radius: 18px; --background-fill-primary: rgba(255,255,255,.06); --background-fill-secondary: rgba(255,255,255,.04); --block-background-fill: rgba(255,255,255,.06); --block-border-color: rgba(255,255,255,.16); --border-color-primary: rgba(255,255,255,.16); --input-background-fill: rgba(255,255,255,.08); --body-text-color: #eef2ff; --body-text-color-subdued: #a9b4d6; --block-label-text-color: #c8d1ee; --block-label-background-fill: rgba(255,255,255,.10); --button-primary-text-color: #fff; --button-primary-background-fill: linear-gradient(135deg, #ff7a59, #ff9a6c); --button-primary-background-fill-hover: linear-gradient(135deg, #ff8c6d, #ffab80); --button-primary-border-color: transparent; }
body { background: radial-gradient(900px 520px at 85% 0%, rgba(30,80,160,.7), transparent 60%), radial-gradient(800px 620px at 0% 100%, rgba(15,110,130,.8), transparent 60%), #0a1330 !important; background-attachment: fixed !important; }
gradio-app, .gradio-container { background: transparent !important; }
.gradio-container { max-width: 1120px !important; margin: auto !important; }
.rtl { direction: rtl; text-align: right; } .small-text { color: #a9b4d6; font-size: .85rem; }
.strip { height: 26px; margin: 4px 0; border-radius: 6px; background: #050b1c repeating-linear-gradient(90deg, #1b2c52 0 22px, transparent 22px 40px) 0 8px / 100% 10px no-repeat; }
.hero { display: flex; align-items: center; justify-content: space-between; gap: 28px; padding: 30px 8px 18px; } .hero-text { flex: 1; } .logo { width: 150px; min-width: 110px; filter: drop-shadow(0 10px 24px rgba(0,0,0,.35)); }
.hero h1 { font-size: 4.2rem; font-weight: 800; line-height: 1.15; margin: 14px 0 4px; color: #fff; } .hero h2 { font-size: 1.25rem; font-weight: 800; margin: 0 0 10px; color: #fff; } .hero p { color: #a9b4d6; line-height: 2; margin: 0 0 18px; max-width: 640px; }
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 8px; } .chip { display: inline-block; padding: 6px 16px; border-radius: 999px; color: #fff; font-weight: 700; font-size: .9rem; } .chip.small { padding: 3px 12px; font-size: .78rem; margin-top: 8px; } .chip.fixed { width: 74px; text-align: center; padding: 3px 0; font-size: .8rem; }
.card, .mood-badge { padding: 14px 18px; border-radius: 18px; margin-top: 12px; background: rgba(255,255,255,.07); border: 1px solid rgba(255,255,255,.16); } .mood-badge { border-inline-start: 5px solid var(--accent); } .mood-name { font-size: 2rem; font-weight: 800; }
.note { margin-top: 8px; padding: 9px 13px; border-radius: 12px; font-size: .88rem; background: rgba(255,255,255,.07); color: #d7defa; } .note.warn { background: #fff1cc; color: #654700; }
.score-row { display: flex; align-items: center; gap: 10px; margin: 8px 0; direction: rtl; } .bar { flex: 1; height: 10px; border-radius: 999px; background: rgba(255,255,255,.12); overflow: hidden; } .bar i { display: block; height: 100%; border-radius: 999px; }
.movie-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(290px, 1fr)); gap: 14px; direction: rtl; } .movie { display: flex; gap: 12px; margin: 0; border-inline-start: 4px solid var(--accent); } .poster { width: 84px; min-width: 84px; height: 120px; object-fit: cover; border-radius: 12px; } .poster.empty { display: flex; align-items: center; justify-content: center; font-size: 2rem; background: linear-gradient(160deg, #ff7a59, #c4452b); } .movie-title { font-size: 1.1rem; font-weight: 800; color: #fff; }
@media (max-width: 700px) { .hero { flex-direction: column-reverse; align-items: flex-start; } .hero h1 { font-size: 3rem; } .logo { width: 96px; } }
'''

genre_choices = [(ALL_GENRES, ALL_GENRES)] + [
    (GENRE_FA.get(g, g), g) for g in sorted({g for m in MOVIES for g in m["genres"]})
]

# This uses the older `source=` argument, which is compatible with Gradio versions
# that warn about `sources=`. Webcam streaming must explicitly use source="webcam".
with gr.Blocks(title="Cinemood | سینمود", css=CSS, theme=gr.themes.Base(), analytics_enabled=False) as demo:
    gr.HTML(HERO)
    with gr.Tabs():
        with gr.Tab("📷 تصویر"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="تصویر چهره", source="upload", type="pil")
                    genre_input = gr.Dropdown(choices=genre_choices, value=ALL_GENRES, label="فیلتر ژانر")
                    crop_input = gr.Checkbox(value=True, label="برش خودکار چهره قبل از تحلیل")
                    run_button = gr.Button("تشخیص حال و پیشنهاد فیلم", variant="primary")
                with gr.Column():
                    mood_html, scores_html = gr.HTML(), gr.HTML()
                    json_output = gr.JSON(label="خروجی قرارداد مشترک")
            results_html = gr.HTML()
            run_button.click(analyze, [image_input, genre_input, crop_input], [json_output, mood_html, scores_html, results_html])

        with gr.Tab("🎥 لایو وب‌کم"):
            with gr.Row():
                with gr.Column():
                    live_cam = gr.Image(label="وب‌کم (تحلیل زنده)", source="webcam", streaming=True, type="numpy")
                    live_genre = gr.Dropdown(choices=genre_choices, value=ALL_GENRES, label="فیلتر ژانر")
                with gr.Column():
                    live_mood, live_scores = gr.HTML(), gr.HTML()
                    live_json = gr.JSON(label="خروجی قرارداد مشترک")
            live_results = gr.HTML()
            live_state = gr.State({"hist": [], "mood": None})
            live_cam.stream(
                analyze_live, [live_cam, live_state, live_genre],
                [live_state, live_json, live_mood, live_scores, live_results],
                show_progress="hidden"
            )
    

if __name__ == "__main__":
    demo.launch(show_api=False)
