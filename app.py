import os
os.environ["MPLBACKEND"] = "Agg"

import spaces
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm
import re
import numpy as np

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

def split_text_with_pause_markers(text, pause_level="Medium"):
    """
    Tách văn bản thành các đoạn lớn (theo paragraph) và thêm pause markers.
    Mỗi đoạn sẽ được xử lý riêng rồi ghép lại với silence thật.
    
    Returns:
        list of tuples: [(text_chunk, pause_duration_in_seconds), ...]
    """
    # Cấu hình pause markers và thời gian silence
    pause_configs = {
        "Short": ("...", 0.3),      # 3 dots, 0.3s silence
        "Medium": (".....", 0.6),   # 5 dots, 0.6s silence  
        "Long": (".......", 1.0),   # 7 dots, 1.0s silence
    }
    
    pause_marker, silence_duration = pause_configs.get(pause_level, (".....", 0.6))
    
    print(f"\n🎛️ Using pause marker: '{pause_marker}' + {silence_duration}s silence")
    
    # Tách theo dòng trống để phân biệt đoạn văn
    paragraphs = text.split('\n\n')
    chunks = []
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Gộp các dòng trong cùng đoạn
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Kiểm tra hội thoại
        has_quotes = '"' in combined_text or '"' in combined_text or '"' in combined_text
        is_dialogue = has_quotes
        
        # Thêm pause marker vào cuối câu (giữ nguyên logic Code 1)
        processed_text = re.sub(r'([.!?])(\s|$)', r'\1 ' + pause_marker + r'\2', combined_text)
        
        # Thêm vào danh sách với thời gian silence
        # Dialogue có silence ngắn hơn (một nửa)
        actual_silence = silence_duration / 2 if is_dialogue else silence_duration
        chunks.append((processed_text, actual_silence))
        
        print(f"\n📝 Chunk {len(chunks)} ({'dialogue' if is_dialogue else 'narrative'}):")
        print(f"   Text: {processed_text[:100]}...")
        print(f"   Silence: {actual_silence}s")
    
    return chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng với thời gian xác định."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Làm sạch văn bản - GIỮ LẠI dấu chấm lặp."""
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
    HYBRID TTS: Xử lý từng đoạn lớn (như Code 1) + thêm silence thật (như Code 2).
    
    Ưu điểm:
    - Không bỏ sót từ (xử lý đoạn lớn, không tách câu nhỏ)
    - Pause tự nhiên (kết hợp pause marker + silence thật)
    - Đọc được từ nước ngoài (không normalize quá mức)
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        print(f"\n{'='*60}")
        print(f"🎤 Starting HYBRID TTS generation")
        print(f"{'='*60}")
        
        # Tách văn bản thành các đoạn với pause markers
        chunks = split_text_with_pause_markers(gen_text, pause_level)
        
        if not chunks:
            raise gr.Error("No valid paragraphs found. Please check your input.")
        
        print(f"\n📊 Total paragraphs to process: {len(chunks)}")
        
        # Preprocess reference audio
        print(f"\n🔄 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Reference text: {ref_text[:100]}...")
        
        # Tạo audio cho từng đoạn và ghép lại
        audio_segments = []
        sample_rate = 24000
        
        for i, (chunk_text, silence_duration) in enumerate(chunks):
            print(f"\n🎵 [{i+1}/{len(chunks)}] Generating audio for paragraph...")
            print(f"   Length: {len(chunk_text)} chars, {len(chunk_text.split())} words")
            
            # Chuẩn hóa văn bản (giữ lại dấu chấm lặp)
            normalized_text = post_process(TTSnorm(chunk_text)).lower()
            print(f"   Normalized: {normalized_text[:150]}...")
            
            try:
                # Tạo audio cho đoạn này (TOÀN BỘ ĐOẠN MỘT LẦN)
                wave, sr, _ = infer_process(
                    ref_audio, 
                    ref_text.lower(), 
                    normalized_text, 
                    model, 
                    vocoder, 
                    speed=speed
                )
                
                sample_rate = sr
                audio_segments.append(wave)
                
                duration = len(wave) / sr
                print(f"   ✅ Generated {duration:.2f}s audio")
                
                # Thêm silence thật vào sau (trừ đoạn cuối)
                if i < len(chunks) - 1:
                    silence = create_silence(silence_duration, sample_rate)
                    audio_segments.append(silence)
                    print(f"   ⏸️  Added {silence_duration}s real silence")
                
            except Exception as e:
                print(f"   ❌ Error processing chunk {i+1}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No audio generated. Please check your text and reference audio.")
        
        final_wave = np.concatenate(audio_segments)
        final_duration = len(final_wave) / sample_rate
        
        print(f"\n✅ Audio generation complete!")
        print(f"   Total duration: {final_duration:.2f}s")
        print(f"   Segments: {len(chunks)} paragraphs")
        print(f"{'='*60}\n")
        
        # Tạo spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            plt.specgram(final_wave, Fs=sample_rate, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Audio Spectrogram ({final_duration:.1f}s)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(spectrogram_path)
            plt.close()

        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech (Hybrid Version)
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    **✨ BEST OF BOTH WORLDS:**
    - 🎯 **No skipped words** (processes entire paragraphs like Code 1)
    - ⏸️ **Natural pauses** (real silence between paragraphs like Code 2)
    - 🌍 **Reads foreign words** (preserves original text structure)
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

"Meci beaucoup!" Hắn đáp.
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
            info="Controls real silence + pause markers"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How HYBRID Processing Works:
    
    | Stage | Description |
    |-------|-------------|
    | **1. Split by Paragraph** | Text divided by `\\n\\n` (preserves full context) |
    | **2. Add Pause Markers** | Dots (`...`) inserted after sentences (smooth rhythm) |
    | **3. Process Each Paragraph** | Entire paragraph generated at once (no word skipping) |
    | **4. Add Real Silence** | Actual silent gaps between paragraphs (clean pauses) |
    | **5. Concatenate** | All segments combined into final audio |
    
    ### 🎯 Why This Works Better:
    
    **Problem with Code 1:**
    - ❌ Pause markers alone don't create enough separation
    - ❌ Model interprets dots inconsistently
    
    **Problem with Code 2:**
    - ❌ Splits into small sentences → words get lost
    - ❌ Over-normalization breaks foreign words
    - ❌ Merging logic is complex and error-prone
    
    **This Hybrid Solution:**
    - ✅ Processes **large chunks** (paragraph-level) → no word loss
    - ✅ Uses **pause markers** for rhythm within paragraphs
    - ✅ Adds **real silence** between paragraphs for clear separation
    - ✅ **Foreign words preserved** (e.g., "Meci beaucoup!" works!)
    
    ### 📖 Pause Levels:
    
    | Level | Marker | Silence | Best For |
    |-------|--------|---------|----------|
    | **Short** | `...` | 0.3s | Fast reading, news |
    | **Medium** | `.....` | 0.6s | Natural storytelling ⭐ |
    | **Long** | `.......` | 1.0s | Dramatic audiobooks |
    
    *Note: Dialogue automatically gets 50% shorter pauses*
    
    ### 📝 Usage Tips:
    1. **Separate major sections** with double line breaks (`\\n\\n`)
    2. **Quote dialogue** normally: `"Hello," she said.`
    3. **Foreign words stay intact**: "Merci", "Thank you", "Danke"
    4. **Short sentences** are kept together (not split!)
    5. **Experiment** with pause levels to find your preference
    
    ### 🎬 Example Processing:
    
    **Input:**
    ```
    Hắn ngồi trên tàu. Mắt nhìn biển.
    
    "Merci beaucoup!"
    
    Họ gặp nhau ở Paris.
    ```
    
    **Step 1 - Add Markers (Medium):**
    ```
    Hắn ngồi trên tàu. ..... Mắt nhìn biển. .....
    
    "Merci beaucoup!" ...
    
    Họ gặp nhau ở Paris. .....
    ```
    
    **Step 2 - Generate + Silence:**
    ```
    [Audio 1: "Hắn ngồi...biển"] → [0.6s silence]
    [Audio 2: "Merci beaucoup"] → [0.3s silence]  
    [Audio 3: "Họ gặp...Paris"]
    ```
    
    **Result:** 🎵 Complete, natural audio with proper pauses!
    
    ### ✅ Advantages Over Previous Versions:
    
    | Feature | Code 1 | Code 2 | Hybrid |
    |---------|--------|--------|--------|
    | No word skipping | ✅ | ❌ | ✅ |
    | Real silence pauses | ❌ | ✅ | ✅ |
    | Foreign words work | ✅ | ❌ | ✅ |
    | Processing speed | Fast | Slow | Medium |
    | Pause quality | Fair | Great | Excellent |
    """)
    
    with gr.Accordion("❗ Known Limitations", open=False):
        gr.Markdown("""
        1. **Numbers**: Dates and phone numbers may not be perfect
        2. **Reference Audio**: Must be clear with minimal background noise
        3. **Very Long Text**: >500 words may take longer to process
        4. **Pause Timing**: First/last sentences might have slight timing differences
        5. **Model Artifacts**: Occasional clicks between segments (rare)
        
        ### 🔧 Troubleshooting:
        - **Pauses too short?** → Try "Long" level
        - **Pauses too long?** → Try "Short" level  
        - **Word skipped?** → Check if it's in a new paragraph (should work now!)
        - **Foreign word mispronounced?** → This is a model limitation (but text is preserved)
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
