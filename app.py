import spaces
import os
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)

# Retrieve token from secrets
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Log in to Hugging Face
if hf_token:
    login(token=hf_token)

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    text = text.replace(" , , ", " , ")
    text = " " + text + " "
    text = text.replace(" ,, ", " , ")
    text = " " + text + " "
    text = text.replace('"', "")
    return " ".join(text.split())

# Load models
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-100h/model_480000.pt")),
    vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-100h/vocab.txt")),
)

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):

    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Please enter text content with less than 100 words.")
    
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, ref_text.lower(), post_process(TTSnorm(gen_text)).lower(), model, vocoder, speed=speed
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis.
    # The model was trained for 480,000 steps with approximately 150 hours of data on an RTX 3090 GPU. 
    Enter text and upload a sample voice to generate natural speech.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(label="📝 Text", placeholder="Enter the text to generate voice...", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
    btn_synthesize = gr.Button("🔥 Generate Voice")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    model_limitations = gr.Textbox(
        value="""1. The model may not perform well with numerical characters, dates, special characters, etc. => A text normalization module is needed.
2. The rhythm of some generated audios may be inconsistent or choppy => It is recommended to select clearly pronounced sample audios with minimal pauses for better synthesis quality.
3. The reference audio text uses the whisper-large-v3-turbo model, which may not always accurately recognize Vietnamese, resulting in poor voice synthesis quality.
4. The current model checkpoint is at around step 480,000, trained with 150 hours of public data => Voice cloning for non-native voices may not be perfectly accurate.
5. Inference with overly long paragraphs may produce poor results.""", 
        label="❗ Model Limitations",
        lines=5,
        interactive=False
    )

    btn_synthesize.click(infer_tts, inputs=[ref_audio, gen_text, speed], outputs=[output_audio, output_spectrogram])

# Run Gradio with share=True to get a gradio.live link
demo.queue().launch()