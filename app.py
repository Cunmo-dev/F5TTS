import os
os.environ["MPLBACKEND"] = "Agg"

import spaces
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm
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

def add_smart_pauses(text, pause_paragraph='...', pause_dialogue='..'):
    """
    Thêm ký tự đặc biệt vào văn bản để tạo khoảng dừng tự nhiên.
    
    Args:
        pause_paragraph: ký tự cho khoảng dừng sau đoạn văn tả (mặc định: '...')
        pause_dialogue: ký tự cho khoảng dừng sau hội thoại (mặc định: '..')
    """
    
    lines = text.split('\n')
    processed_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            processed_lines.append('')
            continue
        
        # Kiểm tra nếu là hội thoại (bắt đầu bằng dấu ngoặc kép)
        is_dialogue_line = line.startswith('"') or line.startswith('"') or line.startswith('"')
        
        if is_dialogue_line:
            # Với hội thoại: thêm pause ngắn sau mỗi câu
            # Tìm các dấu kết thúc câu
            line = re.sub(r'([.!?])\s+', r'\1 ' + pause_dialogue + ' ', line)
            processed_lines.append(line)
        else:
            # Với đoạn văn tả: thêm pause dài hơn
            line = re.sub(r'([.!?])\s+', r'\1 ' + pause_paragraph + ' ', line)
            processed_lines.append(line)
    
    # Ghép lại và làm sạch
    result = '\n'.join(processed_lines)
    
    # Loại bỏ pause thừa ở cuối
    result = re.sub(r'(\.\.\.|\.\.)\s*$', '', result)
    
    return result

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
    ckpt_path=str(cached_path("hf://thanhcong190693/F5TTSVN/model_last.pt")),
    vocab_file=str(cached_path("hf://thanhcong190693/F5TTSVN/config.json")),
)

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, 
              pause_level: str = "Medium", request: gr.Request = None):
    """
    Args:
        pause_level: mức độ khoảng dừng ("Short", "Medium", "Long")
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Chọn ký tự pause dựa trên level
        pause_configs = {
            "Short": (".", ""),           # Pause ngắn: dấu chấm thông thường
            "Medium": ("..", "."),        # Pause vừa: thêm 1-2 dấu chấm
            "Long": ("...", "..")         # Pause dài: thêm 2-3 dấu chấm
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, ("..", "."))
        
        print(f"\n🎛️ Pause config: Paragraph='{pause_paragraph}', Dialogue='{pause_dialogue}'")
        
        # Xử lý văn bản với smart pauses
        processed_text = add_smart_pauses(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Original text length: {len(gen_text)} chars")
        print(f"📝 Processed text length: {len(processed_text)} chars")
        print(f"\n--- PROCESSED TEXT ---")
        print(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
        print("----------------------\n")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Chuẩn hóa văn bản
        normalized_text = post_process(TTSnorm(processed_text)).lower()
        
        print(f"🔄 Normalized text: {normalized_text[:200]}...")
        
        # Tạo audio (XỬ LÝ TOÀN BỘ MỘT LẦN - như code cũ)
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        print("✅ Audio generated successfully!")
        return (final_sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    # The model was trained with approximately 1000 hours of data on a RTX 3090 GPU
    Enter text and upload a sample voice to generate natural speech with intelligent pausing.
    
    ✨ **New Feature**: Smart pause injection - automatically adds natural pauses without splitting sentences!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text", 
            placeholder="Enter the text to generate voice (supports paragraphs and dialogue)...", 
            lines=8
        )
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        pause_level = gr.Radio(
            choices=["Short", "Medium", "Long"],
            value="Medium",
            label="⏸️ Pause Duration",
            info="Short: minimal pauses | Medium: natural pauses | Long: dramatic pauses"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How it works:
    - **Smart Pause Injection**: Automatically detects sentence endings and dialogue
    - **No sentence splitting**: Processes entire text at once (more stable)
    - **Dialogue detection**: Shorter pauses for conversation flow
    - **Paragraph detection**: Longer pauses for narrative text
    
    ### 📊 Pause Levels:
    - **Short**: Quick reading, minimal breaks
    - **Medium**: Natural conversation pace (recommended)
    - **Long**: Dramatic reading, audiobook style
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. This model may not perform well with numerical characters, dates, special characters, etc.
        2. The rhythm of some generated audios may be inconsistent or choppy => Select clearly pronounced sample audios
        3. Reference audio text uses pho-whisper-medium model which may not always accurately recognize Vietnamese
        4. Very long paragraphs (>1000 words) may produce poor results
        """)

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
