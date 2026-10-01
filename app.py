"""Streamlit GUI for neural style transfer.

Run locally:  streamlit run app.py
Also runs as-is on Streamlit Community Cloud (CPU only).
"""
import io
import time
from pathlib import Path

import streamlit as st
from PIL import Image

from style_transfer import load_vgg, pick_device, stylize

st.set_page_config(page_title="Neural Style Transfer", page_icon="🎨", layout="wide")

EXAMPLES = Path(__file__).parent / "examples"


@st.cache_resource(show_spinner="Loading VGG19 (the very first run downloads the model, about a minute)…")
def get_model():
    device = pick_device()
    return load_vgg(device), device


st.title("Neural Style Transfer")
st.caption("Upload a photo and a style image. The photo gets repainted in that style, "
           "using VGG19 features and Gram-matrix style loss (Gatys et al., 2016).")

# ─── Settings ───
with st.sidebar:
    st.header("Settings")
    max_size = st.select_slider("Output size (longest side, px)", options=[192, 256, 320, 384], value=256,
                                help="Bigger looks sharper but takes longer.")
    steps = st.slider("Steps", 50, 600, 200, step=50,
                      help="More steps = stronger, more settled style. 150–300 is usually enough.")
    strength = st.slider("Style strength", 1, 10, 6,
                         help="How much the style overrides the photo's detail.")
    style_weight = 10 ** (strength / 10 * 3 + 4)   # 1 → 10^4.3 … 6 → 10^5.8 … 10 → 10^7

    _, device = get_model()
    if device.type == "cuda":
        st.success("Running on GPU")
    else:
        st.info("Running on CPU: the defaults (256 px, 200 steps) take about 2–4 minutes. "
                "Keep this tab open while it works.")

# ─── Inputs ───
if st.button("✨ Try it with the example images (a puppy + The Great Wave)"):
    st.session_state["use_example"] = True

left, right = st.columns(2)
with left:
    content_file = st.file_uploader("Content image (the photo)", type=["jpg", "jpeg", "png", "webp"])
with right:
    style_file = st.file_uploader("Style image (the painting / texture)", type=["jpg", "jpeg", "png", "webp"])

# Uploads take priority; otherwise fall back to the examples if they were chosen
if content_file or style_file:
    st.session_state["use_example"] = False
use_example = st.session_state.get("use_example", False)
content_src = content_file or (EXAMPLES / "content_puppy.png" if use_example else None)
style_src = style_file or (EXAMPLES / "style_wave.png" if use_example else None)

with left:
    if content_src:
        st.image(str(content_src) if isinstance(content_src, Path) else content_src, width="stretch")
with right:
    if style_src:
        st.image(str(style_src) if isinstance(style_src, Path) else style_src, width="stretch")

run = st.button("Transfer style", type="primary", disabled=not (content_src and style_src), width="stretch")

# ─── Run ───
if run:
    vgg, device = get_model()
    content_img = Image.open(content_src)
    style_img = Image.open(style_src)

    progress = st.progress(0.0, text="Starting…")
    preview_slot = st.empty()
    started = time.time()

    def on_progress(step, loss, preview):
        elapsed = time.time() - started
        eta = elapsed / step * (steps - step)
        progress.progress(step / steps, text=f"Step {step}/{steps} · loss {loss:,.0f} · ~{eta:.0f}s left")
        if preview is not None:
            preview_slot.image(preview, caption=f"Step {step}", width="stretch")

    result = stylize(content_img, style_img, vgg, device, max_size=max_size, steps=steps,
                     style_weight=style_weight, on_progress=on_progress)
    progress.progress(1.0, text=f"Done in {time.time() - started:.0f}s")
    st.session_state["result"] = result

# ─── Result ───
if "result" in st.session_state:
    result = st.session_state["result"]
    st.subheader("Result")
    st.image(result, width="stretch")
    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    st.download_button("Download PNG", buffer.getvalue(), file_name="style_transfer.png", mime="image/png")
