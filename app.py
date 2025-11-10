import spaces
import os
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm
import numpy as np
import re

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

def split_sentences(text):
    """
    Tách văn bản thành các câu dựa trên dấu câu.
    """
    # Tách theo dấu câu: . ! ? và xuống dòng
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    # Lọc bỏ câu rỗng
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def add_silence(audio_array, sample_rate, duration_ms=500):
    """
    Thêm khoảng lặng vào cuối audio array.
    
    Args:
        audio_array: numpy array của audio
        sample_rate: tần số lấy mẫu
        duration_ms: độ dài khoảng lặng (milliseconds)
    """
    silence_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio_array.dtype)
    return np.concatenate([audio_array, silence])

# Load models
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://thanhcong190693/F5TTSVN/model_last.pt")),
    vocab_file=str(cached_path("hf://thanhcong190693/F5TTSVN/config.json")),
)

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, 
              pause_duration: int = 500, request: gr.Request = None):
    """
    Args:
        pause_duration: độ dài khoảng lặng giữa các câu (milliseconds)
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tách văn bản thành các câu
        sentences = split_sentences(gen_text)
        
        if len(sentences) == 0:
            raise gr.Error("No valid sentences found in the text.")
        
        # Khởi tạo danh sách để lưu audio
        audio_segments = []
        sample_rate = None
        all_spectrograms = []
        
        # Xử lý từng câu
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Chuẩn hóa và xử lý câu
            processed_sentence = post_process(TTSnorm(sentence)).lower()
            
            # Tạo audio cho câu
            wave, sr, spectrogram = infer_process(
                ref_audio, 
                ref_text.lower(), 
                processed_sentence, 
                model, 
                vocoder, 
                speed=speed
            )
            
            if sample_rate is None:
                sample_rate = sr
            
            # Thêm khoảng lặng sau mỗi câu (trừ câu cuối)
            if i < len(sentences) - 1:
                wave = add_silence(wave, sr, pause_duration)
            
            audio_segments.append(wave)
            all_spectrograms.append(spectrogram)
        
        # Ghép tất cả audio lại
        final_wave = np.concatenate(audio_segments)
        
        # Tạo spectrogram tổng hợp (lấy spectrogram đầu tiên để hiển thị)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(all_spectrograms[0], spectrogram_path)

        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    # The model was trained with approximately 1000 hours of data on a RTX 3090 GPU
    Enter text and upload a sample voice to generate natural speech with pauses between sentences.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text", 
            placeholder="Enter the text to generate voice...", 
            lines=5
        )
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        pause_duration = gr.Slider(
            100, 2000, value=500, step=100, 
            label="⏸️ Pause Between Sentences (ms)"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 Tips:
    - Adjust **Pause Duration** to control silence between sentences (500ms = 0.5 seconds)
    - For dialogue, use 300-500ms pauses
    - For dramatic reading, use 700-1000ms pauses
    """)
    
    model_limitations = gr.Textbox(
        value="""1. This model may not perform well with numerical characters, dates, special characters, etc. => A text normalization module is needed.
2. The rhythm of some generated audios may be inconsistent or choppy => It is recommended to select clearly pronounced sample audios with minimal pauses for better synthesis quality.
3. Default, reference audio text uses the pho-whisper-medium model, which may not always accurately recognize Vietnamese, resulting in poor voice synthesis quality.
4. Inference with overly long paragraphs may produce poor results.""", 
        label="❗ Model Limitations",
        lines=4,
        interactive=False
    )

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_duration], 
        outputs=[output_audio, output_spectrogram]
    )

# Run Gradio with share=True to get a gradio.live link
demo.queue().launch(share=True)
