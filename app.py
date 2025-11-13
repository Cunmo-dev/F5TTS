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

def is_problematic_text(text):
    """
    Kiểm tra văn bản có vấn đề (lặp từ, chỉ có dấu câu, quá ngắn...)
    """
    # Loại bỏ khoảng trắng và chuyển thường
    clean = text.strip().lower()
    
    # Kiểm tra nếu chỉ có dấu câu và khoảng trắng
    if re.match(r'^[.,!?;:\s]+$', clean):
        return True, "only_punctuation"
    
    # Kiểm tra lặp từ (3+ từ giống nhau liên tiếp)
    words = clean.split()
    if len(words) >= 3:
        # Kiểm tra lặp từ đơn: "há há há"
        if len(set(words)) == 1:
            return True, "repeated_word"
        
        # Kiểm tra pattern lặp: "ha ha ha"
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.4:  # Nếu < 40% từ là unique
            return True, "low_diversity"
    
    # Kiểm tra chỉ có âm thanh không có nghĩa
    sound_patterns = [r'^(ha|he|hi|ho|hu|a|e|i|o|u|ơ|ư|á|à|ả|ã|ạ)+$']
    for pattern in sound_patterns:
        if re.match(pattern, clean.replace(' ', '')):
            return True, "sound_only"
    
    return False, None

def safe_normalize(text):
    """Normalize văn bản an toàn, xử lý các trường hợp đặc biệt."""
    try:
        # Kiểm tra văn bản có vấn đề TRƯỚC khi normalize
        is_problematic, reason = is_problematic_text(text)
        if is_problematic:
            print(f"   ⚠️ Problematic text detected ({reason}): '{text[:50]}...' - WILL BE SKIPPED")
            # Trả về None để báo hiệu cần skip
            return None
        
        # Normalize bình thường
        normalized = TTSnorm(text)
        
        # Kiểm tra kết quả sau normalize
        if len(normalized.strip()) < 2:
            print(f"   ⚠️ Normalized result too short, using original")
            return text.lower()
        
        # Kiểm tra lại sau normalize
        is_prob_after, reason_after = is_problematic_text(normalized)
        if is_prob_after:
            print(f"   ⚠️ Problematic after normalize ({reason_after}) - WILL BE SKIPPED")
            return None
        
        return normalized.lower()
        
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original text")
        return text.lower()

def validate_text_for_tts(text):
    """Kiểm tra và làm sạch văn bản trước khi đưa vào TTS."""
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    # Kiểm tra văn bản có vấn đề
    is_problematic, reason = is_problematic_text(text)
    if is_problematic:
        print(f"   🚫 Invalid text ({reason}): '{text[:50]}...'")
        return None  # Trả về None để skip
    
    # Kiểm tra độ dài tối thiểu
    words = text.split()
    if len(words) < 2:
        print(f"   ⚠️ Too short ({len(words)} words), padding...")
        text = text + " đó nha"
    
    # Kiểm tra độ dài tối đa (tránh câu quá dài)
    if len(words) > 100:
        print(f"   ⚠️ Too long ({len(words)} words), truncating...")
        text = ' '.join(words[:100])
    
    return text

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu, chỉ ghép câu < 2 từ bằng dấu chấm.
    
    Returns:
        list of tuples: [(sentence, pause_duration_in_seconds, is_merged), ...]
        - is_merged: True nếu là câu gộp (đã có dấu chấm nội tại)
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
                    chunks.append((sentence_text, pause_duration, False))
                    current_sentence = ""
        
        # Thêm phần còn lại nếu có
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration, False))
    
    # Gộp các câu < 2 từ bằng dấu chấm
    merged_chunks = []
    temp_sentences = []  # Danh sách các câu tích lũy
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, _) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        
        if word_count >= 3:
            # Câu đủ dài
            if temp_sentences:
                # Gộp các câu tích lũy + câu hiện tại bằng dấu chấm
                all_sentences = temp_sentences + [sentence]
                merged_text = ". ".join(all_sentences)
                # Đánh dấu là câu gộp
                merged_chunks.append((merged_text, pause, True))
                temp_sentences = []
            else:
                # Câu độc lập
                merged_chunks.append((sentence, pause, False))
        else:
            # Câu ngắn (< 2 từ), tích lũy
            temp_sentences.append(sentence)
            temp_pause = pause
            
            # Nếu là câu cuối -> gộp với câu trước
            if is_last:
                if merged_chunks:
                    # Gộp vào câu trước bằng dấu chấm
                    last_sentence, last_pause, last_merged = merged_chunks[-1]
                    combined_text = last_sentence + ". " + ". ".join(temp_sentences)
                    merged_chunks[-1] = (combined_text, last_pause, True)
                    print(f"   🔗 Merged last short chunk(s) with period")
                    temp_sentences = []
                else:
                    # Không có câu trước -> thêm padding
                    merged_text = ". ".join(temp_sentences)
                    while len(merged_text.split()) < 3:
                        merged_text += " này"
                    print(f"   ⚠️ Last chunk too short, padded: '{merged_text}'")
                    merged_chunks.append((merged_text, temp_pause, False))
                    temp_sentences = []
    
    # Xử lý câu còn sót
    if temp_sentences:
        if merged_chunks:
            # Gộp vào câu trước bằng dấu chấm
            last_sentence, last_pause, last_merged = merged_chunks[-1]
            combined_text = last_sentence + ". " + ". ".join(temp_sentences)
            merged_chunks[-1] = (combined_text, last_pause, True)
            print(f"   🔗 Merged remaining short chunks with period")
        else:
            # Trường hợp đặc biệt: chỉ có câu ngắn
            merged_text = ". ".join(temp_sentences)
            while len(merged_text.split()) < 3:
                merged_text += " này"
            print(f"   ⚠️ Only short sentence(s) found, padded: '{merged_text}'")
            merged_chunks.append((merged_text, temp_pause, False))
    
    return merged_chunks

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
        for idx, (sent, pause, is_merged) in enumerate(chunks[:5], 1):
            marker = "🔗 MERGED" if is_merged else "📄 SINGLE"
            print(f"   {idx}. [{marker}, {pause}s] {sent[:80]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        skipped_count = 0
        
        for i, (sentence, pause_duration, is_merged) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:80]}...")
            
            # Chuẩn hóa văn bản an toàn
            normalized_text = safe_normalize(sentence)
            
            # BỎ QUA nếu safe_normalize trả về None (văn bản có vấn đề)
            if normalized_text is None:
                print(f"   ⏭️ SKIPPED: Problematic text pattern detected")
                skipped_count += 1
                # Thêm khoảng im lặng ngắn thay thế
                if i < len(chunks) - 1:
                    silence = create_silence(0.3, sample_rate)
                    audio_segments.append(silence)
                continue
            
            # Post-process
            normalized_text = post_process(normalized_text)
            
            # Validate văn bản - BỎ QUA nếu không hợp lệ
            normalized_text = validate_text_for_tts(normalized_text)
            if normalized_text is None:
                print(f"   ⏭️ SKIPPED: Invalid text after validation")
                skipped_count += 1
                # Thêm khoảng im lặng ngắn thay thế
                if i < len(chunks) - 1:
                    silence = create_silence(0.3, sample_rate)
                    audio_segments.append(silence)
                continue
            
            # Kiểm tra độ dài tối thiểu
            word_count = len(normalized_text.strip().split())
            if word_count < 2:
                print(f"   ⏭️ Skipped (too short: {word_count} words): '{normalized_text}'")
                skipped_count += 1
                continue
            
            print(f"   📝 Normalized ({word_count} words): {normalized_text[:80]}...")
            if is_merged:
                print(f"   ℹ️ Merged sentence - model will create natural pauses at periods")
            
            # Retry logic với backoff
            max_retries = 2
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
                    
                    # Thêm khoảng im lặng giữa các chunk chính
                    if i < len(chunks) - 1 and not is_merged:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Added {pause_duration}s silence between chunks")
                    elif i < len(chunks) - 1 and is_merged:
                        print(f"   🔇 No manual silence (merged sentence with periods)")
                        
                except Exception as e:
                    retry_count += 1
                    error_msg = str(e)[:100]
                    print(f"   ⚠️ Attempt {retry_count} failed: {error_msg}")
                    
                    if retry_count > max_retries:
                        print(f"   ❌ Max retries reached")
                        skipped_count += 1
                        
                        # Thử với văn bản đơn giản hóa
                        if len(normalized_text.split()) > 3:
                            print(f"   🔧 Trying with first 3 words only...")
                            simplified_text = ' '.join(normalized_text.split()[:3])
                            try:
                                wave, sr, _ = infer_process(
                                    ref_audio, 
                                    ref_text.lower(), 
                                    simplified_text, 
                                    model, 
                                    vocoder, 
                                    speed=speed
                                )
                                sample_rate = sr
                                audio_segments.append(wave)
                                print(f"   ✅ Generated with simplified text")
                                success = True
                                skipped_count -= 1  # Không tính là skip
                            except:
                                print(f"   ❌ Simplified attempt also failed, skipping")
                        break
                    
                    # Đợi một chút trước khi retry
                    import time
                    time.sleep(0.5)
        
        # Kiểm tra xem có audio nào được tạo không
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. All sentences were skipped due to problematic patterns (repeated words, sounds only, etc.). Please check your text.")
        
        # Cảnh báo nếu có nhiều câu bị skip
        if skipped_count > 0:
            print(f"\n⚠️ Warning: {skipped_count} chunk(s) were skipped due to problematic patterns")
        
        # Ghép tất cả audio lại
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s (from {len(chunks)-skipped_count}/{len(chunks)} chunks)")
        
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
Hắn lúc này đang ngồi trên boong tàu. Mắt nhìn ra biển xa.

"Toa lần này trở về nhà chơi được bao lâu?"

Người hỏi là một người bạn tình cờ gặp.""", 
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
    ### 💡 How Smart Pause Works (Modified):
    
    | Feature | Description |
    |---------|-------------|
    | **Paragraph Detection** | Separates narrative text by double line breaks |
    | **Dialogue Detection** | Identifies quoted speech (even multi-line) |
    | **Smart Period Merging** | Only sentences < 2 words are merged with periods |
    | **Model-Based Pauses** | AI naturally pauses at periods |
    | **Three Levels** | Short (0.2s/0.1s), Medium (0.4s/0.2s), Long (0.6s/0.3s) |
    | **Problematic Text Skip** | Auto-skips repeated words, sound-only text (not replaced) |
    
    ### 📖 Usage Tips:
    - **Separate paragraphs** with double line breaks (`\n\n`)
    - **Dialogue** can span multiple lines - just use quotes `"..."`
    - **Avoid repeated sounds**: "Há há há", "He he he" will be automatically SKIPPED
    - **Natural prosody**: Model creates pauses at periods
    - **Short**: Fast-paced reading
    - **Medium**: Natural storytelling (recommended)
    - **Long**: Dramatic audiobooks
    
    ### 🎯 Example Processing:
    ```
    Input:
    "Nhà chồng em!"
    "Há há há..."
    "Còn quýt nữa?"
    
    → "Há há há..." is problematic (repeated sound) → SKIPPED entirely
    → Result: Only "Nhà chồng em" and "Còn quýt nữa?" are processed
    ```
    
    ### ⚠️ Note:
    - **Problematic patterns are auto-detected**: Repeated words, sound-only text, only punctuation
    - **These will be either skipped or replaced** with descriptive text
    - Longer texts take more time but produce better pause quality
    - Check console for detailed processing logs
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
        5. **Foreign Words**: Pronounced phonetically in Vietnamese
        6. **Repeated Sounds**: "Há há há", "He he he" etc. will be SKIPPED entirely
        7. **Error Recovery**: If one sentence fails, processing continues with remaining text
        8. **Text Validation**: Problematic patterns (low diversity, sound-only) are filtered out
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
