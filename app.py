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

try:
    from unidecode import unidecode
    from langdetect import detect, LangDetectException
    LANG_DETECT_AVAILABLE = True
except ImportError:
    LANG_DETECT_AVAILABLE = False
    print("⚠️ Warning: langdetect not installed. Foreign word detection disabled.")
    print("   Install with: pip install langdetect unidecode")

def is_vietnamese_char(char):
    """Kiểm tra ký tự có phải tiếng Việt không."""
    vietnamese_chars = set('aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩịoóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵđ')
    vietnamese_chars.update('AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊOÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴĐ')
    return char.lower() in vietnamese_chars or not char.isalpha()

def is_vietnamese_word(word):
    """Kiểm tra từ có phải tiếng Việt không dựa trên tỷ lệ ký tự."""
    clean_word = ''.join(c for c in word if c.isalpha())
    if not clean_word:
        return True  # Số, dấu câu → giữ nguyên
    
    viet_chars = sum(1 for c in clean_word if is_vietnamese_char(c))
    return viet_chars / len(clean_word) >= 0.5  # Ít nhất 50% là ký tự Việt

def transliterate_to_vietnamese(text, lang=None):
    """
    Chuyển văn bản ngoại ngữ sang phiên âm tiếng Việt.
    
    Phương pháp:
    1. Detect ngôn ngữ (nếu chưa biết)
    2. Chuyển sang phiên âm Latin (unidecode)
    3. Điều chỉnh phát âm cho tiếng Việt
    """
    if not text.strip():
        return text
    
    # Phát hiện ngôn ngữ
    if lang is None and LANG_DETECT_AVAILABLE:
        try:
            lang = detect(text)
        except LangDetectException:
            lang = 'unknown'
    
    # Xử lý theo ngôn ngữ
    result = text.lower()
    
    # 1. Chuyển sang phiên âm Latin cơ bản
    if LANG_DETECT_AVAILABLE:
        result = unidecode(result)
    
    # 2. Điều chỉnh phát âm cho tiếng Việt
    replacements = {
        # Tiếng Anh
        'th': 'đ',      # the → đơ
        'ch': 'ch',     # change → chanh
        'sh': 's',      # shop → sốp
        'ph': 'f',      # phone → fôn
        
        # Phụ âm cuối
        'ck': 'c',      # back → bắc
        'ng': 'ng',     # king → kíng
        'tion': 'sơn',  # action → ắc sơn
        
        # Nguyên âm
        'oo': 'u',      # book → búc
        'ee': 'i',      # see → xi
        'ea': 'i',      # tea → ti
        'ou': 'ao',     # house → hao
        'ow': 'ao',     # now → nao
    }
    
    # Áp dụng quy tắc chuyển đổi
    for pattern, replacement in replacements.items():
        result = result.replace(pattern, replacement)
    
    # 3. Xóa các phụ âm cuối khó phát âm
    # k, t, p ở cuối từ → thêm thanh ngắn
    result = re.sub(r'([ktp])(\s|$)', r'\1ơ\2', result)
    
    print(f"   🌍 Transliterated '{text}' → '{result}' (lang: {lang})")
    return result

def process_mixed_language_text(text, mode="transliterate"):
    """
    Xử lý văn bản hỗn hợp nhiều ngôn ngữ.
    
    Args:
        text: Văn bản đầu vào
        mode: "transliterate" (phiên âm), "remove" (xóa), "keep" (giữ nguyên)
    
    Returns:
        Văn bản đã xử lý
    """
    if not LANG_DETECT_AVAILABLE and mode == "transliterate":
        print("⚠️ Langdetect not available, keeping original text")
        return text
    
    words = text.split()
    processed_words = []
    
    for word in words:
        # Giữ nguyên số và dấu câu
        if not any(c.isalpha() for c in word):
            processed_words.append(word)
            continue
        
        # Tách dấu câu
        match = re.match(r'^([^\w]*)([\w]+)([^\w]*)$', word, re.UNICODE)
        if not match:
            processed_words.append(word)
            continue
        
        prefix, core_word, suffix = match.groups()
        
        # Kiểm tra có phải tiếng Việt không
        if is_vietnamese_word(core_word):
            processed_words.append(word)
        else:
            if mode == "transliterate":
                # Chuyển sang phiên âm
                transliterated = transliterate_to_vietnamese(core_word)
                processed_words.append(prefix + transliterated + suffix)
            elif mode == "remove":
                # Xóa từ ngoại ngữ
                print(f"   🚫 Removed foreign word: '{word}'")
                continue
            else:  # keep
                processed_words.append(word)
    
    return ' '.join(processed_words)

def split_text_into_sentences(text, pause_paragraph_duration=0.5, pause_dialogue_duration=0.25, 
                              foreign_word_mode="transliterate"):
    """
    Tách văn bản thành các câu, tự động xử lý từ ngoại ngữ.
    
    Args:
        foreign_word_mode: "transliterate" (phiên âm), "remove" (xóa), "keep" (giữ nguyên)
    """
    # Xử lý từ ngoại ngữ
    text = process_mixed_language_text(text, mode=foreign_word_mode)
    
    chunks = []
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Phát hiện hội thoại
        open_quotes = combined_text.count('"') + combined_text.count('"')
        close_quotes = combined_text.count('"') + combined_text.count('"')
        is_dialogue = (open_quotes > 0 and open_quotes == close_quotes)
        pause_duration = pause_dialogue_duration if is_dialogue else pause_paragraph_duration
        
        # Loại bỏ ngoặc kép
        clean_text = combined_text.replace('"', '').replace('"', '').replace('"', '').strip()
        
        # Tách câu
        sentences = re.split(r'([.!?]+)', clean_text)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:
                current_sentence += part
            else:
                current_sentence += part
                sentence_text = current_sentence.strip()
                
                if sentence_text and len(sentence_text.split()) >= 1:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
                elif sentence_text:
                    current_sentence += " "
        
        if current_sentence.strip() and len(current_sentence.strip().split()) >= 3:
            chunks.append((current_sentence.strip(), pause_duration))
    
    # Gộp câu ngắn
    merged_chunks = []
    temp_sentence = ""
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        
        if word_count >= 5:
            if temp_sentence:
                merged_chunks.append((temp_sentence + " " + sentence, pause))
                temp_sentence = ""
            else:
                merged_chunks.append((sentence, pause))
        else:
            if temp_sentence:
                temp_sentence += " " + sentence
            else:
                temp_sentence = sentence
                temp_pause = pause
            
            should_output = (len(temp_sentence.split()) >= 5) or (is_last and len(temp_sentence.split()) >= 1)
            
            if should_output:
                merged_chunks.append((temp_sentence, temp_pause))
                temp_sentence = ""
    
    if temp_sentence and len(temp_sentence.split()) >= 2:
        merged_chunks.append((temp_sentence, temp_pause))
    
    return merged_chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def apply_fade(audio, fade_samples, fade_type='out'):
    """Áp dụng fade in/out."""
    if len(audio) < fade_samples:
        fade_samples = len(audio)
    
    fade_curve = np.linspace(0, 1, fade_samples) if fade_type == 'in' else np.linspace(1, 0, fade_samples)
    
    if fade_type == 'in':
        audio[:fade_samples] = audio[:fade_samples] * fade_curve
    else:
        audio[-fade_samples:] = audio[-fade_samples:] * fade_curve
    
    return audio

def post_process(text):
    """Làm sạch văn bản."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
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
              pause_level: str = "Medium", cross_fade_duration: float = 0.15,
              foreign_word_mode: str = "Transliterate", request: gr.Request = None):
    """TTS inference với xử lý tự động từ ngoại ngữ."""
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cấu hình pause
        pause_configs = {
            "Short": (0.3, 0.15),
            "Medium": (0.5, 0.25),
            "Long": (0.8, 0.4)
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.5, 0.25))
        
        # Chuyển đổi mode
        mode_map = {
            "Transliterate": "transliterate",
            "Remove": "remove",
            "Keep": "keep"
        }
        mode = mode_map.get(foreign_word_mode, "transliterate")
        
        print(f"\n🎛️ Config: Pause={pause_paragraph}s/{pause_dialogue}s, Foreign={mode}")
        print(f"🔀 Cross-fade: {cross_fade_duration}s")
        
        # Tách văn bản
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue, mode)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks[:3], 1):
            print(f"   {idx}. [{pause}s] {sent[:60]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio
        audio_segments = []
        sample_rate = 24000
        fade_samples = int(cross_fade_duration * sample_rate)
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:60]}...")
            
            try:
                normalized_text = post_process(TTSnorm(sentence)).lower()
            except Exception as e:
                print(f"   ❌ Error normalizing: {e}")
                continue
            
            if len(normalized_text.strip()) < 3:
                print(f"   ⏭️ Skipped (too short): '{normalized_text}'")
                continue
            
            print(f"   📝 Normalized: {normalized_text[:80]}...")
            
            try:
                wave, sr, _ = infer_process(
                    ref_audio, 
                    ref_text.lower(), 
                    normalized_text, 
                    model, 
                    vocoder, 
                    speed=speed,
                    cross_fade_duration=cross_fade_duration
                )
                
                sample_rate = sr
                
                if i < len(chunks) - 1:
                    wave = apply_fade(wave.copy(), fade_samples, 'out')
                
                audio_segments.append(wave)
                print(f"   ✅ Generated {len(wave)/sr:.2f}s audio")
                
                if i < len(chunks) - 1:
                    silence = create_silence(pause_duration, sample_rate)
                    audio_segments.append(silence)
                    print(f"   ⏸️  Added {pause_duration}s silence")
                    
            except Exception as e:
                print(f"   ❌ Error processing: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if not audio_segments:
            raise gr.Error("No valid audio segments generated.")
            
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s")
        
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
    gr.Markdown(f"""
    # 🎤 F5-TTS: Vietnamese TTS with Auto Foreign Word Handling
    
    ### ✨ Features:
    - **🌍 Auto-detect foreign languages** (English, Chinese, Thai, Hindi, etc.)
    - **🔄 Auto-transliterate** to Vietnamese pronunciation
    - **⏸️ Smart pauses** between sentences
    - **🔀 Smooth transitions** with cross-fade
    
    {"✅ **Language detection enabled** (langdetect installed)" if LANG_DETECT_AVAILABLE else "⚠️ **Install langdetect for auto-detection**: `pip install langdetect unidecode`"}
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter mixed language text...

Tiếng Việt:
Hắn ngồi trên boong tàu. "Hello, how are you?" hắn hỏi.

English + French:
The weather is nice. "Merci beaucoup!" he said.

中文 + ไทย:
你好世界。สวัสดีครับ。""", 
            lines=10
        )
    
    with gr.Row():
        speed = gr.Slider(0.5, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        pause_level = gr.Radio(
            choices=["Short", "Medium", "Long"],
            value="Medium",
            label="⏸️ Pause Duration"
        )
    
    with gr.Row():
        cross_fade = gr.Slider(
            0.05, 0.3, value=0.15, step=0.05, 
            label="🔀 Cross-fade (s)"
        )
        foreign_word_mode = gr.Radio(
            choices=["Transliterate", "Remove", "Keep"],
            value="Transliterate",
            label="🌍 Foreign Words",
            info="How to handle non-Vietnamese words"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 🌍 Foreign Word Modes:
    
    | Mode | Description | Example |
    |------|-------------|---------|
    | **Transliterate** ✅ | Convert to Vietnamese sound | "Hello" → "hê lô" |
    | **Remove** | Delete foreign words | "Hello world" → "world" (if Vietnamese) |
    | **Keep** | Keep original (may fail) | "Hello" → model error |
    
    ### 🔄 Auto-transliteration Examples:
    
    ```
    English:  "Hello world"     → "hê lô oa đơ"
    French:   "Merci beaucoup"  → "mẹc xi bô cu"
    Chinese:  "你好" (nǐ hǎo)   → "ni hao"
    Thai:     "สวัสดี" (sawatdi)→ "sawatdi"
    Hindi:    "नमस्ते" (namaste)→ "namaste"
    ```
    
    ### 💡 How It Works:
    1. **Detect** language of each word (langdetect)
    2. **Transliterate** to Latin script (unidecode)
    3. **Adjust** pronunciation for Vietnamese TTS
    4. **Generate** natural speech
    
    ### 📦 Installation (for full features):
    ```bash
    pip install langdetect unidecode
    ```
    
    ### ⚠️ Limitations:
    - Transliteration is phonetic approximation, not perfect
    - Complex foreign phrases may sound unnatural
    - Best for simple foreign words/names
    """)

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level, cross_fade, foreign_word_mode], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
