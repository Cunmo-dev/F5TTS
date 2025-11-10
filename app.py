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

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tach van ban thanh cac cau, xu ly dung hoi thoai tung dong.
    
    Returns:
        list of tuples: [(sentence, pause_duration_in_seconds), ...]
    """
    chunks = []
    
    # Tach theo dong don de xu ly tung dong rieng biet
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Bo qua cac dong chi chua ky tu dac biet
        special_chars_only = re.match(r'^[\+\-\*=_#@&%\$!~`\|/\\]+$', line)
        if special_chars_only:
            continue
        
        # Kiem tra xem dong nay co phai la hoi thoai khong (co dau ngoac kep)
        has_quotes = ('"' in line or '"' in line or '"' in line)
        pause_duration = pause_dialogue_duration if has_quotes else pause_paragraph_duration
        
        # Loai bo dau ngoac kep va ky tu + de xu ly
        clean_text = line.replace('"', '').replace('"', '').replace('"', '').replace('+', '').strip()
        
        # Bo qua neu sau khi lam sach khong con gi
        if not clean_text:
            continue
        
        # Tach thanh cac cau dua tren dau cau
        sentences = re.split(r'([.!?]+)', clean_text)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Phan van ban
                current_sentence += part
            else:  # Dau cau
                current_sentence += part
                sentence_text = current_sentence.strip()
                
                # Chi them neu cau khong rong
                if sentence_text:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
        
        # Them phan con lai neu co
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    return chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tao doan im lang voi thoi gian xac dinh."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Lam sach van ban."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('+', "")
    # Loai bo dau phay du thua
    text = re.sub(r',+', ',', text)
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
    TTS inference voi pause thuc su bang cach ghep audio.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cau hinh pause (giay)
        pause_configs = {
            "Short": (0.4, 0.2),
            "Medium": (0.8, 0.4),
            "Long": (1.2, 0.6)
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.8, 0.4))
        
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tach van ban thanh cac cau voi thoi gian dung
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks[:5], 1):
            print(f"   {idx}. [{pause}s] {sent[:60]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tao audio cho tung cau va ghep lai
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:60]}...")
            
            # Chuan hoa van ban
            normalized_text = post_process(TTSnorm(sentence)).lower()
            
            # Kiem tra do dai
            if len(normalized_text.strip()) < 2:
                print(f"   ⏭️ Skipped (too short after normalize): '{normalized_text}'")
                continue
            
            print(f"   📝 Normalized: {normalized_text[:80]}...")
            
            try:
                # Tao audio cho cau nay
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
                print(f"   ✅ Generated {len(wave)/sr:.2f}s audio")
                
                # Them khoang im lang (tru cau cuoi)
                if i < len(chunks) - 1:
                    silence = create_silence(pause_duration, sample_rate)
                    audio_segments.append(silence)
                    print(f"   ⏸️  Added {pause_duration}s silence")
                    
            except Exception as e:
                print(f"   ❌ Error processing chunk: {e}")
                continue
        
        # Ghep tat ca audio lai
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text.")
            
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s ({len(audio_segments)//2} segments)")
        
        # Tao spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            plt.specgram(final_wave, Fs=sample_rate, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Audio Spectrogram')
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
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    Enter text and upload a sample voice to generate natural speech with **real silence pauses**.
    
    ✨ **Smart Pause Feature**: Automatically adds REAL silent pauses between sentences!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs and dialogue...

Example:
Thiếu nữ cười rộ lên, trong nắng chiều tịch mịch nàng tựa như tiên tử thoát thai chốn hồng trần.
"Cái này em cho cậu. Cậu cứ ăn đi đừng ngại."
"Vậy..."
Minh Huy hơi khó xử nhìn nàng, bất quá hắn cũng không quá khách sáo.
"Vậy được... cảm ơn em!"
Hắn liền lột vỏ một trái rồi cắn xuống.""", 
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
            info="Controls REAL silence duration between sentences"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How Smart Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Line-by-Line Processing** | Each line is processed separately |
    | **Dialogue Detection** | Automatically detects quoted speech with `"..."` |
    | **Real Silence** | Actual silent gaps (no fake sounds!) |
    | **Three Levels** | Short (0.4s/0.2s), Medium (0.8s/0.4s), Long (1.2s/0.6s) |
    
    ### 📖 Usage Tips:
    - **Each line** is treated as a separate unit
    - **Dialogue lines** (with quotes `"..."`) get shorter pauses
    - **Narrative lines** (without quotes) get longer pauses
    - **Short**: Fast-paced reading (news, announcements)
    - **Medium**: Natural storytelling (recommended)
    - **Long**: Dramatic audiobooks, poetry
    
    ### 🎯 Example Input:
    ```
    Thiếu nữ cười rộ lên, trong nắng chiều tịch mịch.
    "Cái này em cho cậu. Cậu cứ ăn đi đừng ngại."
    "Vậy..."
    Minh Huy hơi khó xử nhìn nàng.
    "Vậy được... cảm ơn em!"
    ```
    
    **Result**: Each dialogue line gets 0.4s pause, narrative lines get 0.8s pause (Medium setting)
    
    ### ⚠️ Note:
    - Each sentence is processed separately, then combined with real silence
    - Longer texts take more time but produce better pause quality
    - Dialogue lines are automatically detected by quotes
    - All sentences are processed individually without merging
    - Lines with only special characters (+, -, *, etc.) are automatically skipped
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
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
