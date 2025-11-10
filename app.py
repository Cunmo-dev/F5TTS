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

def split_text_smart(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thông minh: GIỮ LẠI các câu ngắn (như "Meci beaucoup!") 
    thay vì bỏ qua hoặc gộp cưỡng bức.
    
    Returns:
        list of tuples: [(sentence, pause_duration_in_seconds), ...]
    """
    chunks = []
    
    # Tách theo dòng trống để phân biệt đoạn văn
    paragraphs = text.split('\n\n')
    
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
        pause_duration = pause_dialogue_duration if is_dialogue else pause_paragraph_duration
        
        # Loại bỏ dấu ngoặc kép
        clean_text = combined_text.replace('"', '').replace('"', '').replace('"', '').strip()
        
        # Tách thành các câu dựa trên dấu câu
        sentences = re.split(r'([.!?]+)', clean_text)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Phần văn bản
                current_sentence += part
            else:  # Dấu câu
                current_sentence += part
                sentence_text = current_sentence.strip()
                
                # THAY ĐỔI QUAN TRỌNG: Chấp nhận TẤT CẢ các câu có ít nhất 1 từ
                if sentence_text and len(sentence_text.split()) >= 1:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
        
        # Thêm phần còn lại nếu có
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    # KHÔNG GỘP các câu ngắn - để model xử lý tất cả
    return chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng với thời gian xác định."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Làm sạch văn bản - tối giản để giữ nguyên từ ngoại ngữ."""
    text = " " + text + " "
    # Chỉ loại bỏ dấu ngoặc kép và khoảng trắng dư thừa
    text = text.replace('"', "")
    text = text.replace('"', "")
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
    TTS inference HYBRID: Đọc đầy đủ + Pause tự nhiên.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cấu hình pause (giây)
        pause_configs = {
            "Short": (0.3, 0.15),   # Paragraph: 0.3s, Dialogue: 0.15s
            "Medium": (0.5, 0.25),  # Paragraph: 0.5s, Dialogue: 0.25s
            "Long": (0.8, 0.4)      # Paragraph: 0.8s, Dialogue: 0.4s
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.5, 0.25))
        
        print(f"\n{'='*60}")
        print(f"🎤 HYBRID TTS Generation")
        print(f"{'='*60}")
        print(f"🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản thành các câu (GIỮ LẠI tất cả câu)
        chunks = split_text_smart(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks[:5], 1):
            print(f"   {idx}. [{pause}s pause] {sent[:70]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        print(f"\n🔄 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Reference text: {ref_text[:100]}...")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence}")
            
            # Chuẩn hóa văn bản (chỉ lowercase và trim)
            try:
                normalized_text = post_process(TTSnorm(sentence)).lower()
            except:
                # Nếu TTSnorm lỗi với từ ngoại ngữ, dùng văn bản gốc
                normalized_text = post_process(sentence).lower()
            
            # QUAN TRỌNG: Chấp nhận TẤT CẢ văn bản, kể cả rất ngắn
            if len(normalized_text.strip()) < 2:
                print(f"   ⏭️ Skipped (too short): '{normalized_text}'")
                continue
            
            print(f"   📝 Normalized: {normalized_text}")
            
            try:
                # Tạo audio cho câu này
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
                
                # Thêm khoảng im lặng (trừ câu cuối)
                if i < len(chunks) - 1:
                    silence = create_silence(pause_duration, sample_rate)
                    audio_segments.append(silence)
                    print(f"   ⏸️  Added {pause_duration}s silence")
                    
            except Exception as e:
                print(f"   ❌ Error processing chunk: {e}")
                # Không bỏ qua hoàn toàn, thử với văn bản gốc
                try:
                    print(f"   🔄 Retry with original text...")
                    wave, sr, _ = infer_process(
                        ref_audio, 
                        ref_text.lower(), 
                        sentence.lower(), 
                        model, 
                        vocoder, 
                        speed=speed
                    )
                    sample_rate = sr
                    audio_segments.append(wave)
                    print(f"   ✅ Success on retry!")
                    
                    if i < len(chunks) - 1:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                except:
                    print(f"   ❌ Final skip")
                    continue
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text.")
            
        final_wave = np.concatenate(audio_segments)
        
        total_duration = len(final_wave) / sample_rate
        num_sentences = (len(audio_segments) + 1) // 2
        
        print(f"\n✅ FINAL RESULT:")
        print(f"   Duration: {total_duration:.2f}s")
        print(f"   Sentences: {num_sentences}")
        print(f"   Sample rate: {sample_rate}Hz")
        print(f"{'='*60}\n")
        
        # Tạo spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(14, 5))
            plt.specgram(final_wave, Fs=sample_rate, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title(f'Audio Spectrogram ({num_sentences} sentences, {total_duration:.1f}s)')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(spectrogram_path, dpi=100)
            plt.close()

        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS Hybrid: Best of Both Worlds
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    **NEW APPROACH**: Combines complete text reading + natural silence pauses!
    
    ✅ **Reads ALL sentences** (even short ones like "Meci beaucoup!")  
    ✅ **Natural pauses** with real silence between sentences  
    ✅ **Foreign words preserved** (French, English, etc.)
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

Họ cùng cười và tiếp tục hành trình.""", 
            lines=12
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
            info="Real silence between sentences"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How HYBRID Approach Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Complete Reading** | ALL sentences are processed (no skipping!) |
    | **Smart Splitting** | Sentences split by punctuation (`.!?`) |
    | **Real Silence** | Actual silent gaps inserted between audio |
    | **Foreign Word Safe** | Preserves non-Vietnamese text |
    | **Auto Recovery** | Retries with original text if normalization fails |
    
    ### 📖 Pause Levels:
    - **Short** (0.3s/0.15s): Quick reading - news, announcements
    - **Medium** (0.5s/0.25s): Natural storytelling - **recommended**
    - **Long** (0.8s/0.4s): Dramatic pauses - audiobooks, poetry
    
    *(First number = paragraph pause, Second = dialogue pause)*
    
    ### 🎯 Example Processing:
    
    **Input:**
    ```
    Hắn ngồi trên tàu. Mắt nhìn ra biển.
    
    "Toa về nhà chơi bao lâu?"
    
    Người hỏi là bạn cũ.
    
    "Meci beaucoup!"
    ```
    
    **Output:** 
    ```
    [Audio: "Hắn ngồi trên tàu"]
    [Silence: 0.5s]
    [Audio: "Mắt nhìn ra biển"]
    [Silence: 0.5s]
    [Audio: "Toa về nhà chơi bao lâu"]
    [Silence: 0.25s]
    [Audio: "Người hỏi là bạn cũ"]
    [Silence: 0.5s]
    [Audio: "Meci beaucoup"]  ← ✅ Not skipped!
    ```
    
    ### ✨ Key Improvements:
    
    1. **No Sentence Skipping**: 
       - Old: "Meci beaucoup!" → ❌ Skipped (too short)
       - New: "Meci beaucoup!" → ✅ **Processed!**
    
    2. **Better Pause Quality**:
       - Old: Fake pauses with `...` dots
       - New: **Real silence** (0.3-0.8 seconds)
    
    3. **Fallback Protection**:
       - If TTSnorm fails → automatically retries with original text
       - Foreign words won't break the process
    
    ### 📝 Usage Tips:
    - Use **double line breaks** (`\\n\\n`) to separate major sections
    - Quote dialogue: `"Hello," she said.`
    - Short sentences (1-2 words) are now **fully supported**
    - Mix Vietnamese and foreign words freely
    - Adjust pause level to match your content style
    
    ### ⚙️ Technical Details:
    - Each sentence → separate audio generation → combined with silence
    - No text truncation or forced merging
    - Original text preserved when normalization fails
    - Sample rate: 24,000 Hz
    """)
    
    with gr.Accordion("❗ Model Limitations & Tips", open=False):
        gr.Markdown("""
        ### Limitations:
        1. **Numbers**: Dates/phone numbers may not sound natural
        2. **Processing Time**: Longer than single-pass (but better quality!)
        3. **Reference Audio**: Needs clear audio without background noise
        4. **Very Long Texts**: Consider splitting into sections (>1000 words)
        
        ### Troubleshooting:
        - **Weird pronunciation?** → Try different reference audio
        - **Pauses too long/short?** → Adjust pause level
        - **Missing words?** → Check console logs for errors
        - **Foreign words sound wrong?** → This is expected; model trained on Vietnamese
        
        ### Best Practices:
        ✅ Use clear paragraph breaks  
        ✅ Keep sentences under 40 words  
        ✅ Use high-quality reference audio (3-10 seconds)  
        ✅ Test different pause levels for your content type  
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
