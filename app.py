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

import os
from huggingface_hub import login

# Lấy token từ secrets
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Login vào Hugging Face
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
    return " ".join(text.split())

# Load models
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://hynt/F5-TTS-Vietnamese-100h/model_350000.pt")),
    vocab_file=str(cached_path("hf://hynt/F5-TTS-Vietnamese-100h/vocab.txt")),
)

def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, request: gr.Request = None):

    if not ref_audio_orig:
        raise gr.Error("Vui lòng tải lên tệp âm thanh mẫu.")
    if not gen_text.strip():
        raise gr.Error("Vui lòng nhập nội dung cần sinh giọng.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Vui lòng nhập nội dung cần sinh giọng nhỏ hơn 100 từ.")
    
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, ref_text, post_process(TTSnorm(gen_text)), model, vocoder, speed=speed
        )
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"Lỗi khi sinh giọng: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Tổng hợp giọng nói Tiếng Việt.
    # Mô hình được huấn luyện 350.000 steps với bộ dữ liệu khoảng 100h trên 1 GPU RTX 3090. 
    Nhập văn bản và tải lên một mẫu giọng để tạo âm thanh tự nhiên.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Mẫu giọng", type="filepath")
        gen_text = gr.Textbox(label="📝 Văn bản", placeholder="Nhập nội dung cần sinh giọng...", lines=3)
    
    speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Tốc độ")
    btn_synthesize = gr.Button("🔥 Sinh giọng")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Âm thanh tạo ra", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    model_limitations = gr.Textbox(
        value="""1. Mô hình có thể hoạt động không tốt với các ký tự số, ngày tháng, ký tự đặc biệt, ... => cần bổ sung thêm một module text normalization (chuẩn hoá text).
2. Nhịp điệu của một số audio có thể chưa được mạch lạc, giật cục.
3. Audio reference text sử dụng model whisper-large-v3-turbo nên sẽ có một vài trường hợp không nhận diện chính xác Tiếng Việt, dẫn đến kết quả tổng hợp giọng nói rất tệ.
4. Checkpoint của mô hình hiện tại dừng lại ở khoảng step thứ 350.000, được huấn luyện với 100 giờ dữ liệu public.""", 
        label="❗ Hạn chế của mô hình",
        lines=4,
        interactive=False
    )

    btn_synthesize.click(infer_tts, inputs=[ref_audio, gen_text, speed], outputs=[output_audio, output_spectrogram])

# Chạy Gradio với share=True để có link gradio.live
demo.queue().launch()