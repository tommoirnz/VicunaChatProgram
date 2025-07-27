# WebLLAVA1.py  – 2025‑07‑27 (mirror‑corrected)
# -*- coding: utf-8 -*-
"""
LLaVA‑13B multimodal assistant (Gradio 4.x)
------------------------------------------
* Model: llava‑hf/llava‑v1.6‑vicuna‑13b‑hf (4‑bit NF4)
* Features: text + image chat, Whisper ASR, pyttsx3 TTS, HTTPS Gradio UI.
"""

import sys, asyncio, os, datetime, tempfile, torch, sounddevice as sd, soundfile as sf
from typing import List, Dict, Any, Optional
import PIL.Image as Image
import gradio as gr
import whisper, pyttsx3

# ───────────────────────  Environment  ───────────────────────
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["USE_TORCH_COMPILE"] = "0"

# ───────────────────────  Load LLaVA  ────────────────────────
from transformers import BitsAndBytesConfig
from transformers.models.llava_next.processing_llava_next import LlavaNextProcessor
from transformers.models.llava_next.modeling_llava_next import LlavaNextForConditionalGeneration

MODEL_ID    = "llava-hf/llava-v1.6-vicuna-13b-hf"
MODEL_CACHE = r"C:\models"

qcfg = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

processor = LlavaNextProcessor.from_pretrained(
    MODEL_ID, cache_dir=MODEL_CACHE, trust_remote_code=True
)
llava = LlavaNextForConditionalGeneration.from_pretrained(
    MODEL_ID, device_map="auto",
    quantization_config=qcfg, torch_dtype=torch.float16,
    cache_dir=MODEL_CACHE, trust_remote_code=True
)

# ───────────────────────  Whisper & TTS  ─────────────────────
whisper_mod = whisper.load_model("small", device="cpu")

def record_audio(duration: int, fs: int = 16000) -> str:
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype="float32")
    sd.wait()
    path = tempfile.mktemp(suffix=".wav")
    sf.write(path, audio, fs)
    return path

def transcribe(path: str) -> str:
    try:
        return whisper_mod.transcribe(path, task="translate")["text"].strip()
    except Exception as e:
        return f"Whisper error: {e}"
    finally:
        if os.path.exists(path):
            os.remove(path)

def voices() -> List[str]:
    engine = pyttsx3.init("sapi5")
    return [f"{i}: {v.name}" for i, v in enumerate(engine.getProperty("voices"))]

def speak(txt: str, idx: str, rate: int = 200) -> str:
    engine = pyttsx3.init("sapi5")
    vs = engine.getProperty("voices")
    engine.setProperty("voice", vs[int(idx)].id)
    engine.setProperty("rate", rate)
    out = tempfile.mktemp(suffix=".wav")
    engine.save_to_file(txt, out)
    engine.runAndWait()
    return out

# ───────────────────────  Chat helpers  ─────────────────────
SYSTEM_PROMPT = "You are a helpful, concise AI assistant."
hist: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]

def reset_history() -> tuple[str, str, Optional[str], Optional[Image.Image]]:
    hist.clear()
    hist.append({"role": "system", "content": SYSTEM_PROMPT})
    return "", "", None, None

def shortcut(msg: str) -> Optional[str]:
    text = (msg or "").lower()
    if "date" in text and any(k in text for k in ("today", "current", "what")):
        return datetime.date.today().strftime("Today's date is %B %d, %Y.")
    return None

def chat(message: str, image: Optional[Image.Image] = None) -> str:
    if not message or not message.strip():
        return ""
    if reply := shortcut(message):
        hist.append({"role": "assistant", "content": reply})
        return reply

    entry = ([{"type": "image", "image": image}, {"type": "text", "text": message}]
             if image is not None else [{"type": "text", "text": message}])

    prompt_history = hist + [{"role": "user", "content": entry}]
    prompt_text = processor.apply_chat_template(
        prompt_history, add_generation_prompt=True, tokenize=False
    )
    inputs = processor(
        text=prompt_text,
        images=image if image is not None else None,
        return_tensors="pt", padding=True
    ).to(DEVICE)

    with torch.no_grad():
        out_ids = llava.generate(**inputs, max_new_tokens=512)

    decoded = processor.decode(out_ids[0], skip_special_tokens=True)
    reply = decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded.strip()

    import re
    reply = re.sub(r'^(user|assistant):\s*', '', reply, flags=re.IGNORECASE)
    reply = re.sub(r'\b(user|assistant):', '', reply, flags=re.IGNORECASE).strip()

    hist.extend([{"role": "user", "content": message},
                 {"role": "assistant", "content": reply}])
    return reply

# ───────────────────────  UI  ───────────────────────────────
CSS = """
textarea, input, .gradio-container {font-size:24px !important;}
button.btn-small {min-width:120px !important;}
#webcam-preview video, #webcam-preview img { transform: scaleX(-1); }   /* mirror fix */
"""

def build_ui():
    with gr.Blocks(css=CSS) as demo:
        gr.Markdown("<h1 align='center'>🖼️ LLaVA‑13B Assistant</h1>")

        # Text entry, Send, Record
        with gr.Row():
            txt  = gr.Textbox(lines=3, label="Your message", scale=8)
            send = gr.Button("Send", variant="primary", elem_classes="btn-small", scale=1)
            rec  = gr.Button("🎤 Record", elem_classes="btn-small", scale=1)

        # Recording duration slider
        with gr.Row():
            dur_slider = gr.Slider(5, 20, value=5, step=1,
                                   label="Recording duration (seconds)", scale=10)

        # Image tabs
        with gr.Row():
            with gr.Tab("Upload"):
                img_upload = gr.Image(
                    sources=["upload"], type="pil", label="Upload"
                )

            with gr.Tab("Webcam"):
                img_cam = gr.Image(
                    sources=["webcam"], type="pil", label="Webcam",
                    elem_id="webcam-preview"          # tie CSS rule to this element
                )

            img_state = gr.State()

            # live preview → state (with un‑mirror)
            def unmirror(frame: Image.Image | None):
                return frame.transpose(Image.FLIP_LEFT_RIGHT) if frame else None

            img_upload.change(lambda x: x, img_upload, img_state)
            img_cam.change(unmirror, img_cam, img_state)

        # Controls
        with gr.Row():
            auto     = gr.Checkbox(True, label="Auto‑TTS")
            voice_dd = gr.Dropdown(choices=voices(), value=voices()[0], label="Voice")
            reset    = gr.Button("🔄 New topic")

        # Outputs
        with gr.Row():
            md_out  = gr.Markdown()
            aud_out = gr.Audio(type="filepath", autoplay=True, label="TTS")

        # Pipeline glue
        def pipeline(msg, img, do_tts, voice):
            if not (msg and msg.strip()) and img is None:
                return "", None, None
            reply = chat(msg, img)
            wav   = speak(reply, int(voice.split(":")[0])) if do_tts else None
            return reply, wav, None

        send.click(pipeline, [txt, img_state, auto, voice_dd],
                   [md_out, aud_out, img_state])
        txt.submit(pipeline, [txt, img_state, auto, voice_dd],
                   [md_out, aud_out, img_state])

        rec.click(lambda d: transcribe(record_audio(int(d))),
                  dur_slider, txt)

        reset.click(reset_history, None, [md_out, txt, aud_out, img_state])

    return demo

# ───────────────────────  Launch  ───────────────────────────
demo = build_ui()

if __name__ == "__main__":
    os.environ["GRADIO_SSL_NO_VERIFY"] = "1"
    demo.launch(
        server_name="0.0.0.0", server_port=8443,
        ssl_certfile="C:/Users/tomsp/Downloads/cert.pem",
        ssl_keyfile="C:/Users/tomsp/Downloads/key.pem",
        ssl_verify=False, share=True
    )
