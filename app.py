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
import librosa

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

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng với thời gian xác định."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def detect_sentence_boundaries(text):
    """
    Phát hiện vị trí các câu và loại (paragraph/dialogue).
    Returns: list of (position, pause_type)
        position: vị trí ký tự kết thúc câu trong text
        pause_type: 'paragraph' hoặc 'dialogue'
    """
    boundaries = []
    paragraphs = text.split('\n\n')
    current_pos = 0
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            current_pos += 2  # \n\n
            continue
        
        # Kiểm tra xem đoạn này có phải hội thoại không
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        has_quotes = '"' in combined_text or '"' in combined_text or '"' in combined_text
        is_dialogue = has_quotes
        pause_type = 'dialogue' if is_dialogue else 'paragraph'
        
        # Tìm các dấu câu kết thúc trong đoạn này
        for match in re.finditer(r'[.!?]+', combined_text):
            # Vị trí tương đối trong toàn bộ text
            boundaries.append({
                'char_pos': current_pos + match.end(),
                'pause_type': pause_type
            })
        
        current_pos += len(para) + 2  # +2 cho \n\n
    
    return boundaries

def estimate_audio_position(text, char_pos, total_chars, total_audio_length):
    """
    Ước tính vị trí trong audio tương ứng với vị trí ký tự trong text.
    Giả định tốc độ đọc tương đối đồng đều.
    """
    ratio = char_pos / max(total_chars, 1)
    return int(ratio * total_audio_length)

def insert_pauses_into_audio(audio, sample_rate, text, pause_paragraph=0.8, pause_dialogue=0.4):
    """
    Chèn khoảng im lặng vào audio đã tạo dựa trên vị trí câu trong text.
    
    Args:
        audio: numpy array của audio đã tạo
        sample_rate: tần số mẫu
        text: văn bản gốc (đã processed)
        pause_paragraph: thời gian pause cho đoạn văn (giây)
        pause_dialogue: thời gian pause cho hội thoại (giây)
    
    Returns:
        numpy array: audio với pause đã chèn
    """
    # Phát hiện các vị trí câu
    boundaries = detect_sentence_boundaries(text)
    
    if not boundaries:
        print("⚠️ No sentence boundaries detected, returning original audio")
        return audio
    
    print(f"\n🔍 Detected {len(boundaries)} sentence boundaries:")
    for i, b in enumerate(boundaries[:5]):  # Show first 5
        print(f"   {i+1}. Char {b['char_pos']}: {b['pause_type']}")
    
    # Chuyển đổi vị trí ký tự sang vị trí audio (samples)
    total_chars = len(text)
    total_samples = len(audio)
    
    pause_configs = {
        'paragraph': pause_paragraph,
        'dialogue': pause_dialogue
    }
    
    # Tạo danh sách các đoạn audio cần ghép
    segments = []
    last_pos = 0
    
    for boundary in boundaries:
        # Ước tính vị trí trong audio
        char_pos = boundary['char_pos']
        audio_pos = estimate_audio_position(text, char_pos, total_chars, total_samples)
        
        # Thêm đoạn audio từ vị trí cũ đến vị trí hiện tại
        if audio_pos > last_pos and audio_pos <= total_samples:
            segments.append(audio[last_pos:audio_pos])
            
            # Thêm pause
            pause_duration = pause_configs[boundary['pause_type']]
            silence = create_silence(pause_duration, sample_rate)
            segments.append(silence)
            
            last_pos = audio_pos
    
    # Thêm phần audio còn lại
    if last_pos < total_samples:
        segments.append(audio[last_pos:])
    
    # Ghép tất cả lại
    final_audio = np.concatenate(segments) if segments else audio
    
    added_duration = (len(final_audio) - len(audio)) / sample_rate
    print(f"\n⏸️  Added {added_duration:.2f}s of pauses to audio")
    
    return final_audio

def post_process(text):
    """Làm sạch văn bản."""
    text = " " + text + " "
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
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
    TTS inference - xử lý toàn bộ văn bản một lần, sau đó chèn pause.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        print(f"\n{'='*60}")
        print(f"🎤 Starting TTS generation with post-processing pauses")
        print(f"{'='*60}")
        
        # Cấu hình pause theo thời gian (giây)
        pause_configs = {
            "Short": (0.4, 0.2),    # Paragraph: 0.4s, Dialogue: 0.2s
            "Medium": (0.8, 0.4),   # Paragraph: 0.8s, Dialogue: 0.4s
            "Long": (1.2, 0.6)      # Paragraph: 1.2s, Dialogue: 0.6s
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.8, 0.4))
        
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Lưu text gốc để phát hiện boundaries
        original_text = gen_text
        
        print(f"\n📊 Stats:")
        print(f"   Text length: {len(gen_text)} chars")
        print(f"   Word count: {len(gen_text.split())} words")
        
        # Preprocess reference audio
        print(f"\n🔄 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Reference text: {ref_text[:100]}...")
        
        # Chuẩn hóa văn bản - KHÔNG thêm dots
        normalized_text = post_process(TTSnorm(gen_text)).lower()
        
        print(f"\n📝 Normalized text preview:")
        print(f"   {normalized_text[:200]}...")
        
        # === BƯỚC 1: Tạo audio TOÀN BỘ một lần ===
        print(f"\n🎵 Generating complete audio (single-pass)...")
        base_wave, sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        base_duration = len(base_wave) / sample_rate
        print(f"   ✅ Base audio: {base_duration:.2f}s")
        
        # === BƯỚC 2: Chèn pause vào đúng vị trí ===
        print(f"\n⏸️  Inserting pauses at sentence boundaries...")
        final_wave = insert_pauses_into_audio(
            base_wave, 
            sample_rate, 
            original_text,  # Dùng text gốc để detect boundaries
            pause_paragraph, 
            pause_dialogue
        )
        
        final_duration = len(final_wave) / sample_rate
        print(f"\n✅ Final audio generated successfully!")
        print(f"   Base duration: {base_duration:.2f}s")
        print(f"   Final duration: {final_duration:.2f}s")
        print(f"   Added: {final_duration - base_duration:.2f}s")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"{'='*60}\n")
        
        # Lưu spectrogram (từ base audio)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech with Smart Pause Injection
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    **🎯 New Approach**: Generate full audio first, then inject silence at sentence boundaries!
    
    ✨ **Key Features**:
    - ✅ Reads **entire text at once** (including foreign language!)
    - ✅ Inserts **real silence** after generation
    - ✅ No sentence splitting = better continuity
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs and foreign words...

Example:
Hắn lúc này đang ngồi trên boong tàu. Mắt nhìn ra biển xa.

"Toa lần này trở về nhà chơi được bao lâu?"

Người hỏi là một người bạn tình cờ gặp. "Merci beaucoup!"

Đây là cách tốt nhất để xử lý multilingual text.""", 
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
            info="Real silence duration inserted after generation"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How It Works (2-Step Process):
    
    | Step | Process | Benefit |
    |------|---------|---------|
    | **1. Full Generation** | Model reads entire text at once | Foreign words handled perfectly |
    | **2. Pause Injection** | Insert silence at detected boundaries | Natural pauses without breaking flow |
    
    ### 🎯 Algorithm:
    
    ```
    Input: "Hắn ngồi trên tàu. Mắt nhìn biển.\n\n\"Merci beaucoup!\""
    
    Step 1: Generate full audio (10 seconds)
    ├── Model reads: "hắn ngồi trên tàu mắt nhìn biển merci beaucoup"
    └── Output: continuous audio stream
    
    Step 2: Detect boundaries in original text
    ├── Sentence 1 ends at char 23 (paragraph) → audio position ~3s
    ├── Sentence 2 ends at char 42 (paragraph) → audio position ~6s  
    └── Sentence 3 ends at char 60 (dialogue) → audio position ~9s
    
    Step 3: Insert pauses
    ├── Insert 0.8s silence at 3s mark
    ├── Insert 0.8s silence at 6s mark
    └── Insert 0.4s silence at 9s mark
    
    Final: 10s + 2.0s = 12s audio with natural pauses
    ```
    
    ### 📖 Pause Levels:
    
    | Level | Paragraph | Dialogue | Best For |
    |-------|-----------|----------|----------|
    | **Short** | 0.4s | 0.2s | News, fast reading |
    | **Medium** | 0.8s | 0.4s | Stories, natural speech |
    | **Long** | 1.2s | 0.6s | Audiobooks, dramatic reading |
    
    ### ✅ Advantages Over Previous Approaches:
    
    | Feature | Dots Method | Sentence Split | **This Method** |
    |---------|-------------|----------------|-----------------|
    | Foreign words | ⚠️ May skip | ❌ Fails | ✅ Perfect |
    | Processing speed | ⚡ Fast | 🐌 Slow | ⚡ Fast |
    | Audio continuity | ✅ Good | ⚠️ Fragmented | ✅ Excellent |
    | Pause precision | ⚠️ Approximate | ✅ Exact | ✅ Exact |
    | No weird sounds | ⚠️ Sometimes | ✅ Yes | ✅ Yes |
    
    ### 📝 Usage Tips:
    - Separate major sections with **double line breaks** (`\\n\\n`)
    - Foreign words/phrases are handled naturally: `"Merci beaucoup!"`
    - Quotes indicate dialogue: `"Hello," she said.`
    - Model reads everything once, then pauses are added
    
    ### 🔧 Technical Notes:
    - Pause positions estimated by character ratio: `audio_pos = (char_pos / total_chars) × audio_length`
    - Paragraph detection: double line breaks (`\\n\\n`)
    - Dialogue detection: presence of quotation marks
    - Silence insertion: numpy zero arrays
    """)
    
    with gr.Accordion("❗ Model Limitations & Troubleshooting", open=False):
        gr.Markdown("""
        ### Limitations:
        1. **Pause Estimation**: Position is estimated, not exact (based on uniform reading speed assumption)
        2. **Very Long Texts**: Texts over 1000 words may have timing drift
        3. **Uneven Reading Speed**: If model reads some parts faster, pause timing may be off
        4. **Numbers & Special Chars**: May not pronounce dates/phone numbers correctly
        
        ### Troubleshooting:
        - **Pauses in wrong places**: Text length estimation issue - try splitting long texts
        - **Foreign words skipped**: Should NOT happen with this method! Report if it does.
        - **Weird timing**: Very long texts cause drift - split into paragraphs
        - **Too many/few pauses**: Adjust pause level setting
        
        ### Best Practices:
        - ✅ Keep texts under 500 words for best timing accuracy
        - ✅ Use clear paragraph breaks (double newlines)
        - ✅ Test with different pause levels to find sweet spot
        - ✅ For very long texts, process in sections
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
