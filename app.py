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
hf_token = os.getenv("HUGGINGFACAHUB_API_TOKEN")

# Log in to Hugging Face
if hf_token:
    login(token=hf_token)

def convert_time_to_dots(pause_seconds, dots_per_second=10):
    """
    Chuyển đổi thời gian pause (giây) thành số lượng dots.
    
    Args:
        pause_seconds: Thời gian pause mong muốn (giây)
        dots_per_second: Hệ số chuyển đổi (mặc định: 10 dots = 1 giây)
    
    Returns:
        str: Chuỗi dots tương ứng
    """
    num_dots = max(2, int(pause_seconds * dots_per_second))  # Tối thiểu 2 dots
    return "." * num_dots

def add_natural_pauses(text, pause_level="Medium"):
    """
    Thêm ký hiệu đặc biệt để tạo khoảng dừng tự nhiên.
    Sử dụng thời gian (giây) để tính số dots tương ứng.
    """
    # Cấu hình pause theo thời gian (giây)
    pause_time_configs = {
        "Short": (0.4, 0.2),    # Paragraph: 0.4s, Dialogue: 0.2s
        "Medium": (0.8, 0.4),   # Paragraph: 0.8s, Dialogue: 0.4s
        "Long": (1.2, 0.6)      # Paragraph: 1.2s, Dialogue: 0.6s
    }
    
    pause_paragraph_time, pause_dialogue_time = pause_time_configs.get(
        pause_level, (0.8, 0.4)
    )
    
    # Chuyển đổi thời gian thành dots
    # Công thức: 1 giây ≈ 5 dots (có thể điều chỉnh)
    dots_per_second = 5
    pause_paragraph = convert_time_to_dots(pause_paragraph_time, dots_per_second)
    pause_dialogue = convert_time_to_dots(pause_dialogue_time, dots_per_second)
    
    print(f"\n🎛️ Pause configuration:")
    print(f"   Level: {pause_level}")
    print(f"   Paragraph: {pause_paragraph_time}s → '{pause_paragraph}' ({len(pause_paragraph)} dots)")
    print(f"   Dialogue: {pause_dialogue_time}s → '{pause_dialogue}' ({len(pause_dialogue)} dots)")
    
    # Tách theo dòng trống để phân biệt đoạn văn
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Gộp các dòng trong cùng đoạn
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Kiểm tra hội thoại (có dấu ngoặc)
        has_quotes = '"' in combined_text or '"' in combined_text or '"' in combined_text
        is_dialogue = has_quotes
        
        pause_marker = pause_dialogue if is_dialogue else pause_paragraph
        
        # Thay thế dấu câu cuối bằng dấu câu + pause marker
        # Ví dụ: "Xin chào." -> "Xin chào. ....."
        combined_text = re.sub(r'([.!?])(\s|$)', r'\1 ' + pause_marker + r'\2', combined_text)
        
        processed_paragraphs.append(combined_text)
    
    result = '\n\n'.join(processed_paragraphs)
    
    print(f"\n📝 Processed text preview:")
    print(result[:300] + "..." if len(result) > 300 else result)
    
    return result

def post_process(text):
    """Làm sạch văn bản - GIỮ LẠI dấu chấm lặp để tạo pause."""
    text = " " + text + " "
    # KHÔNG gộp dấu chấm lặp - để model tự xử lý
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    # Chỉ gộp dấu phẩy dư thừa
    text = re.sub(r',+', ',', text)
    text = text.replace(" , ", " ")
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
    TTS inference - xử lý toàn bộ văn bản một lần với pause markers.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        print(f"\n{'='*60}")
        print(f"🎤 Starting TTS generation")
        print(f"{'='*60}")
        
        # Thêm pause markers vào văn bản
        processed_text = add_natural_pauses(gen_text, pause_level)
        
        print(f"\n📊 Stats:")
        print(f"   Original length: {len(gen_text)} chars")
        print(f"   Processed length: {len(processed_text)} chars")
        print(f"   Word count: {len(processed_text.split())} words")
        
        # Preprocess reference audio
        print(f"\n🔄 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Reference text: {ref_text[:100]}...")
        
        # Chuẩn hóa văn bản (giữ lại dấu chấm lặp)
        normalized_text = post_process(TTSnorm(processed_text)).lower()
        
        print(f"\n📝 Normalized text preview:")
        print(f"   {normalized_text[:200]}...")
        
        # Tạo audio - XỬ LÝ TOÀN BỘ MỘT LẦN
        print(f"\n🎵 Generating audio...")
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        duration = len(final_wave) / final_sample_rate
        print(f"\n✅ Audio generated successfully!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {final_sample_rate}Hz")
        print(f"{'='*60}\n")
        
        # Lưu spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    Enter text and upload a sample voice to generate natural speech with **time-based intelligent pause control**.
    
    ✨ **Smart Pause Feature**: Converts pause time (seconds) to dot markers automatically!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs separated by blank lines...

Example:
Hắn lúc này đang ngồi trên boong tàu. Mắt nhìn ra biển xa.

"Toa lần này trở về nhà chơi được bao lâu?"

Người hỏi là một người bạn tình cờ gặp.

"Meci beaucoup!"
""", 
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
            info="Controls natural pauses after sentences (time-based)"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How Time-Based Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Time Configuration** | Set pause duration in seconds (e.g., 0.4s, 0.8s) |
    | **Auto Conversion** | Converts time to dots (1 second ≈ 5 dots) |
    | **Paragraph vs Dialogue** | Different pause times for narrative and speech |
    | **Single-Pass Processing** | Entire text processed at once (no splitting!) |
    | **Consistent Logic** | Same dot-based approach, just smarter configuration |
    
    ### 📖 Pause Levels (Time → Dots):
    
    | Level | Paragraph | Dialogue | Use Case |
    |-------|-----------|----------|----------|
    | **Short** | 0.4s (2 dots) | 0.2s (1 dot) | News, fast reading |
    | **Medium** | 0.8s (4 dots) | 0.4s (2 dots) | Stories, audiobooks |
    | **Long** | 1.2s (6 dots) | 0.6s (3 dots) | Dramatic reading |
    
    ### 🎯 Example Processing:
    
    **Input (with Medium level):**
    ```
    Hắn ngồi trên tàu. Mắt nhìn ra biển.
    
    "Xin chào!"
    ```
    
    **After Time-Based Conversion:**
    ```
    Paragraph (0.8s → 4 dots):
    Hắn ngồi trên tàu. .... Mắt nhìn ra biển. ....
    
    Dialogue (0.4s → 2 dots):
    "Xin chào!" ..
    ```
    
    ### ✅ Advantages:
    - ⏱️ **Intuitive Configuration** - think in seconds, not dots!
    - 🎵 **Natural Rhythm** - automatic conversion ensures consistency
    - ⚡ **Fast Processing** - single pass through model
    - 🔧 **Easy Adjustment** - change `dots_per_second` to fine-tune
    - 📊 **Predictable Results** - time-based settings are easier to understand
    
    ### 🔧 Technical Details:
    - **Conversion Formula**: `num_dots = pause_seconds × dots_per_second`
    - **Default Rate**: 5 dots/second (adjustable in code)
    - **Minimum**: Always at least 2 dots to ensure pause effect
    
    ### 📝 Usage Tips:
    - Separate major sections with **double line breaks** (`\\n\\n`)
    - Quote dialogue: `"Hello," she said.`
    - Short sentences are automatically handled (no skipping!)
    - Experiment with pause levels to find what sounds best
    - If pauses too long/short, adjust `dots_per_second` in code
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not handle dates, phone numbers perfectly
        2. **Audio Quality**: Use clear reference audio with minimal background noise
        3. **Reference Text**: Auto-transcribed using Whisper (may have errors)
        4. **Very Long Text**: Texts over 1000 words may produce inconsistent results
        5. **Foreign Words**: May not pronounce non-Vietnamese words correctly
        6. **Dot Calibration**: The 5 dots/second default may need adjustment based on model behavior
        
        ### 🔧 Troubleshooting:
        - If pauses too long: decrease `dots_per_second` (try 3-4)
        - If pauses too short: increase `dots_per_second` (try 6-8)
        - If you hear weird sounds, use "Short" pause level
        - For very long texts, consider splitting into multiple generations
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
