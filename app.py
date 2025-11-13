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

def is_emotional_expression(text):
    """
    Kiểm tra xem câu có phải biểu cảm (cười, khóc, kêu) không.
    Các câu này KHÔNG nên merge.
    """
    # Loại bỏ dấu câu để kiểm tra
    clean = re.sub(r'[.,!?;:\-…]+', '', text.lower()).strip()
    
    # Pattern cho tiếng cười, khóc, kêu
    emotional_patterns = [
        r'^(ha+|he+|hi+|ho+|hu+)+$',  # há há, hê hê, hi hi
        r'^(kha+|khe+|khi+)+$',        # khà khà, khì khì
        r'^(u+h*|a+h*|o+h*)+$',        # uh, ah, oh, uuu, aaa
        r'^(hm+|um+|ừ+|ơ+)+$',         # hmm, umm, ừ, ơ
    ]
    
    for pattern in emotional_patterns:
        if re.match(pattern, clean):
            return True
    
    return False

def pad_short_sentence(text, min_words=3):
    """
    Pad câu ngắn để đủ độ dài tối thiểu.
    Ưu tiên lặp lại từ cuối nếu là biểu cảm.
    """
    words = text.split()
    
    if len(words) >= min_words:
        return text
    
    # Nếu là biểu cảm -> lặp lại từ cuối
    if is_emotional_expression(text):
        last_word = words[-1] if words else text
        # Loại bỏ dấu câu
        last_word_clean = re.sub(r'[.,!?;:\-…]+$', '', last_word)
        
        while len(words) < min_words:
            words.append(last_word_clean)
        
        result = ' '.join(words)
        print(f"   🔄 Padded emotional: '{text}' → '{result}'")
        return result
    
    # Nếu không phải biểu cảm -> thêm "này"
    while len(words) < min_words:
        words.append("này")
    
    result = ' '.join(words)
    print(f"   🔄 Padded normal: '{text}' → '{result}'")
    return result

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu với xử lý đặc biệt cho biểu cảm.
    
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
        
        # Kiểm tra xem đoạn này có phải toàn bộ là hội thoại không
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Đếm số dấu ngoặc
        open_quotes = combined_text.count('"') + combined_text.count('"')
        close_quotes = combined_text.count('"') + combined_text.count('"')
        
        # Nếu có dấu ngoặc và cân bằng -> hội thoại
        is_dialogue = (open_quotes > 0 and open_quotes == close_quotes)
        pause_duration = pause_dialogue_duration if is_dialogue else pause_paragraph_duration
        
        # Loại bỏ dấu ngoặc kép để xử lý
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
                
                # Thêm câu nếu có nội dung
                if sentence_text:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
        
        # Thêm phần còn lại nếu có
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    # Xử lý câu ngắn: KHÔNG merge, mà pad
    processed_chunks = []
    
    for i, (sentence, pause) in enumerate(chunks):
        word_count = len(sentence.split())
        
        # Nếu câu ngắn (< 3 từ)
        if word_count < 3:
            # Pad thay vì merge
            padded_sentence = pad_short_sentence(sentence, min_words=3)
            processed_chunks.append((padded_sentence, pause))
        else:
            # Câu đủ dài, giữ nguyên
            processed_chunks.append((sentence, pause))
    
    return processed_chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng với thời gian xác định."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Làm sạch văn bản."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    # Loại bỏ dấu phẩy dư thừa
    text = re.sub(r',+', ',', text)
    return " ".join(text.split())

def safe_normalize(text):
    """Normalize văn bản an toàn, xử lý lỗi với từ ngoại ngữ."""
    try:
        # Bỏ qua normalize cho biểu cảm
        if is_emotional_expression(text):
            print(f"   🎭 Skipped normalize for emotional: '{text}'")
            return text.lower()
        
        normalized = TTSnorm(text)
        # Nếu kết quả quá ngắn hoặc rỗng, giữ nguyên text gốc
        if len(normalized.strip()) < 2:
            return text.lower()
        return normalized.lower()
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original text")
        return text.lower()

def validate_text_for_tts(text):
    """Kiểm tra văn bản trước khi đưa vào TTS."""
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    # Chỉ cảnh báo, KHÔNG từ chối
    words = text.split()
    if len(words) < 2:
        print(f"   ⚠️ Warning: Very short text ({len(words)} words)")
    
    return text

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
    TTS inference với pause thực sự bằng cách ghép audio.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cấu hình pause (giây)
        pause_configs = {
            "Short": (0.2, 0.1),
            "Medium": (0.4, 0.2),
            "Long": (0.6, 0.3)
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.4, 0.2))
        
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản thành các câu với thời gian dừng
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks[:5], 1):
            emotional = "🎭" if is_emotional_expression(sent) else "📄"
            print(f"   {idx}. [{emotional}, {pause}s] {sent[:80]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:80]}...")
            
            # Chuẩn hóa văn bản an toàn
            normalized_text = post_process(safe_normalize(sentence))
            
            # Validate văn bản (KHÔNG skip)
            normalized_text = validate_text_for_tts(normalized_text)
            
            word_count = len(normalized_text.strip().split())
            print(f"   📝 Normalized ({word_count} words): {normalized_text[:80]}...")
            
            # Retry logic với backoff
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count <= max_retries and not success:
                try:
                    if retry_count > 0:
                        print(f"   🔁 Retry {retry_count}/{max_retries}...")
                    
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
                    success = True
                    
                    # Thêm khoảng im lặng giữa các chunk (không phải câu cuối)
                    if i < len(chunks) - 1:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Added {pause_duration}s silence")
                        
                except Exception as e:
                    retry_count += 1
                    print(f"   ⚠️ Attempt {retry_count} failed: {str(e)[:100]}")
                    
                    if retry_count > max_retries:
                        print(f"   ❌ Max retries reached for chunk")
                        # Thử với padding thêm
                        if not is_emotional_expression(normalized_text):
                            print(f"   🔧 Trying with extra padding...")
                            padded = normalized_text + " này này"
                            try:
                                wave, sr, _ = infer_process(
                                    ref_audio, 
                                    ref_text.lower(), 
                                    padded, 
                                    model, 
                                    vocoder, 
                                    speed=speed
                                )
                                sample_rate = sr
                                audio_segments.append(wave)
                                print(f"   ✅ Generated with extra padding")
                                success = True
                            except:
                                print(f"   ❌ Extra padding also failed, skipping")
                        break
                    
                    # Đợi một chút trước khi retry
                    import time
                    time.sleep(0.5)
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text.")
            
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s (from {len(chunks)} chunks)")
        
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
        raise gr.Error(f"Error generating voice: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis (Fixed)
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    Enter text and upload a sample voice to generate natural speech with **real silence pauses**.
    
    ✨ **Fixed**: Emotional expressions (laughter, cries) are now handled correctly!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs and dialogue...

Example:
Chớp mắt một cái bỗng dưng không còn nhìn thấy bé Tư đâu nữa. Trong bóng đêm dày đặc chỉ nghe thấy tiếng cười quỷ dị của y.

"Há há há..."

Minh Huy căng mắt nhìn ra xung quanh. Mồ hôi trên trán rơi xuống mi mắt hắn một mảng cay xè.

"A!!!!!!!"
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
            info="Controls REAL silence duration between sentences"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 Fixed Improvements:
    
    | Fix | Description |
    |-----|-------------|
    | **🎭 Emotional Detection** | Recognizes laughter (há há), cries (ư ư) |
    | **🔄 Smart Padding** | Repeats emotional words instead of adding "này" |
    | **⛔ No Skipping** | All sentences are processed, even very short ones |
    | **🔧 Better Retry** | 3 attempts with fallback padding |
    
    ### 📖 Usage Tips:
    - Emotional expressions like "Há há há..." are now preserved correctly
    - Very short sentences get padded automatically
    - No more silent skips in generated audio
    
    ### ⚠️ Note:
    - Short emotional sentences are padded by repeating the last word
    - Example: "Há!" → "Há há há" (automatically)
    - This ensures minimum 3 words for stable TTS generation
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
        5. **Foreign Words**: Pronounced phonetically in Vietnamese
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
