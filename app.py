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

def add_smart_pauses(text, pause_paragraph=', , , ,', pause_dialogue=', ,'):
    """
    Thêm dấu phẩy để tạo khoảng dừng tự nhiên trong TTS.
    
    Args:
        pause_paragraph: dấu phẩy cho pause sau đoạn văn tả (mặc định: ', , , ,')
        pause_dialogue: dấu phẩy cho pause sau hội thoại (mặc định: ', ,')
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
            # Với hội thoại: thêm pause ngắn
            line = re.sub(r'([.!?])\s+', r'\1 ' + pause_dialogue + ' ', line)
            # Xử lý dấu cuối câu không có khoảng trống sau
            line = re.sub(r'([.!?])$', r'\1 ' + pause_dialogue, line)
            processed_lines.append(line)
        else:
            # Với đoạn văn tả: thêm pause dài
            line = re.sub(r'([.!?])\s+', r'\1 ' + pause_paragraph + ' ', line)
            # Xử lý dấu cuối câu không có khoảng trống sau
            line = re.sub(r'([.!?])$', r'\1 ' + pause_paragraph, line)
            processed_lines.append(line)
    
    result = '\n'.join(processed_lines)
    
    # Loại bỏ pause thừa ở cuối văn bản
    result = re.sub(r'[,\s]+$', '', result)
    
    return result

def post_process(text):
    """Làm sạch văn bản nhưng giữ lại dấu phẩy lặp"""
    text = " " + text + " "
    # KHÔNG gộp dấu phẩy lặp - giữ nguyên để tạo pause
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    # Chỉ gộp 3+ dấu phẩy liên tiếp thành 2 dấu phẩy
    text = re.sub(r',(\s*,){3,}', ', ,', text)
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
    TTS inference với smart pause injection.
    
    Args:
        pause_level: "Short", "Medium", hoặc "Long"
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Chọn cấu hình pause dựa trên level
        pause_configs = {
            "Short": (", ,", ","),              # Pause ngắn: 2 vs 1 comma
            "Medium": (", , ,", ", ,"),         # Pause vừa: 3 vs 2 commas
            "Long": (", , , , ,", ", , ,")      # Pause dài: 5 vs 3 commas
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (", , ,", ", ,"))
        
        print(f"\n🎛️ Pause config: Paragraph='{pause_paragraph}', Dialogue='{pause_dialogue}'")
        
        # Thêm smart pauses vào văn bản
        processed_text = add_smart_pauses(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Original text length: {len(gen_text)} chars")
        print(f"📝 Processed text length: {len(processed_text)} chars")
        print(f"\n--- PROCESSED TEXT ---")
        print(processed_text[:500] + "..." if len(processed_text) > 500 else processed_text)
        print("----------------------\n")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Chuẩn hóa văn bản (giữ lại dấu phẩy lặp)
        normalized_text = post_process(TTSnorm(processed_text)).lower()
        
        print(f"🔄 Normalized text preview: {normalized_text[:200]}...")
        
        # Tạo audio (XỬ LÝ TOÀN BỘ MỘT LẦN)
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        # Lưu spectrogram
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
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    Enter text and upload a sample voice to generate natural speech with **intelligent pause control**.
    
    ✨ **Smart Pause Feature**: Automatically adds natural pauses between sentences and dialogue!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="Enter text with paragraphs and dialogue...\n\nExample:\nHe walked slowly. The sun was setting.\n\n\"How are you?\" she asked.\n\n\"I'm fine,\" he replied.", 
            lines=10
        )
    
    with gr.Row():
        speed = gr.Slider(
            minimum=0.3, 
            maximum=2.0, 
            value=1.0, 
            step=0.1, 
            label="⚡ Speed"
        )
        pause_level = gr.Radio(
            choices=["Short", "Medium", "Long"],
            value="Medium",
            label="⏸️ Pause Duration",
            info="Controls silence between sentences and dialogue"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How Smart Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Automatic Detection** | Distinguishes between narrative text and dialogue |
    | **Paragraph Pauses** | Longer pauses after descriptive sentences (`.` `!` `?`) |
    | **Dialogue Pauses** | Shorter pauses between conversation lines |
    | **Three Levels** | Short (quick), Medium (natural), Long (dramatic) |
    
    ### 📖 Usage Tips:
    - **Short**: Best for fast-paced reading, news, announcements
    - **Medium**: Recommended for stories, articles, general content
    - **Long**: Ideal for audiobooks, dramatic readings, poetry
    - Use double line breaks to separate major sections
    - Put dialogue in quotes: `"Hello," he said.`
    
    ### 🎯 Example Input:
    ```
    The old man sat by the river. He watched the boats pass.
    
    "Beautiful day, isn't it?" asked a stranger.
    
    "Indeed it is," the old man replied with a smile.
    ```
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not handle dates, phone numbers well
        2. **Audio Quality**: Use clear reference audio with minimal background noise
        3. **Reference Text**: Auto-transcribed using Whisper (may have errors)
        4. **Long Text**: Very long paragraphs (1000+ words) may produce inconsistent results
        5. **Foreign Words**: May not pronounce non-Vietnamese words correctly
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
