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

def detect_repeated_pattern(text):
    """
    Phát hiện pattern lặp lại như "Há há há", "A a a", "hehe he he"
    Returns: (is_repeated, cleaned_text)
    """
    text_clean = text.strip()
    words = text_clean.split()
    
    # Nếu có 2 từ trở lên
    if len(words) >= 2:
        # Loại bỏ dấu câu để so sánh
        words_normalized = [re.sub(r'[^\w]', '', w.lower()) for w in words]
        
        # Kiểm tra nếu tất cả từ giống nhau (hoặc rất giống nhau)
        unique_words = set(words_normalized)
        
        # Pattern lặp: tất cả từ giống nhau hoặc chỉ có 1-2 từ unique
        if len(unique_words) <= 2 and len(words) >= 2:
            # Kiểm tra xem có phải lặp hoàn toàn không
            first_word = words_normalized[0]
            if all(w == first_word or w == '' for w in words_normalized):
                # Lặp hoàn toàn: "há há há" -> giữ lại 1 lần
                return True, words[0]
            
            # Kiểm tra lặp xen kẽ: "há há hả" (2 từ giống nhau trở lên)
            if len([w for w in words_normalized if w == first_word]) >= len(words) * 0.6:
                return True, words[0]
    
    return False, text_clean

def merge_repeated_with_context(chunks):
    """
    Gộp các chunk có pattern lặp vào câu bên cạnh, hoặc bỏ qua nếu không thể gộp.
    """
    merged = []
    i = 0
    
    while i < len(chunks):
        sentence, pause, is_merged = chunks[i]
        is_repeated, cleaned = detect_repeated_pattern(sentence)
        
        if is_repeated:
            print(f"   🔁 Detected repeated pattern: '{sentence}' -> '{cleaned}'")
            
            # Thử gộp với câu trước
            if merged:
                prev_sentence, prev_pause, prev_merged = merged[-1]
                # Gộp vào câu trước với dấu phẩy
                merged[-1] = (f"{prev_sentence}, {cleaned}", prev_pause, True)
                print(f"   ✅ Merged with previous: '{prev_sentence}' + '{cleaned}'")
            
            # Nếu không có câu trước, thử gộp với câu sau
            elif i + 1 < len(chunks):
                next_sentence, next_pause, next_merged = chunks[i + 1]
                # Gộp vào câu sau
                merged.append((f"{cleaned}, {next_sentence}", next_pause, True))
                print(f"   ✅ Merged with next: '{cleaned}' + '{next_sentence}'")
                i += 1  # Skip câu sau vì đã gộp
            
            # Nếu không gộp được, bỏ qua chunk này
            else:
                print(f"   ⏭️ Skipped standalone repeated pattern: '{sentence}'")
        else:
            # Câu bình thường, thêm vào
            merged.append((sentence, pause, is_merged))
        
        i += 1
    
    return merged

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu, xử lý pattern lặp lại.
    
    Returns:
        list of tuples: [(sentence, pause_duration_in_seconds, is_merged), ...]
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
    
    # XỬ LÝ PATTERN LẶP LẠI TRƯỚC KHI GỘP CÂU NGẮN
    chunks = merge_repeated_with_context(chunks)
    
    # Gộp các câu < 2 từ bằng dấu chấm (logic cũ)
    merged_chunks = []
    temp_sentences = []
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, is_merged_flag) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        
        if word_count >= 3:
            # Câu đủ dài
            if temp_sentences:
                all_sentences = temp_sentences + [sentence]
                merged_text = ". ".join(all_sentences)
                merged_chunks.append((merged_text, pause, True))
                temp_sentences = []
            else:
                merged_chunks.append((sentence, pause, is_merged_flag))
        else:
            # Câu ngắn (< 3 từ), tích lũy
            temp_sentences.append(sentence)
            temp_pause = pause
            
            if is_last:
                if merged_chunks:
                    last_sentence, last_pause, last_merged = merged_chunks[-1]
                    combined_text = last_sentence + ". " + ". ".join(temp_sentences)
                    merged_chunks[-1] = (combined_text, last_pause, True)
                    print(f"   🔗 Merged last short chunk(s) with period")
                    temp_sentences = []
                else:
                    merged_text = ". ".join(temp_sentences)
                    while len(merged_text.split()) < 3:
                        merged_text += " này"
                    print(f"   ⚠️ Last chunk too short, padded: '{merged_text}'")
                    merged_chunks.append((merged_text, temp_pause, False))
                    temp_sentences = []
    
    # Xử lý câu còn sót
    if temp_sentences:
        if merged_chunks:
            last_sentence, last_pause, last_merged = merged_chunks[-1]
            combined_text = last_sentence + ". " + ". ".join(temp_sentences)
            merged_chunks[-1] = (combined_text, last_pause, True)
            print(f"   🔗 Merged remaining short chunks with period")
        else:
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
    """Làm sạch văn bản, xử lý dấu chấm liên tiếp."""
    text = " " + text + " "
    
    # Xử lý dấu chấm liên tiếp (... -> .)
    text = re.sub(r'\.{2,}', '.', text)
    
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    
    # Loại bỏ dấu phẩy dư thừa
    text = re.sub(r',+', ',', text)
    
    # Loại bỏ dấu chấm than/hỏi liên tiếp quá nhiều (!!! -> !)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)
    
    return " ".join(text.split())

def safe_normalize(text):
    """Normalize văn bản an toàn, xử lý lỗi với từ ngoại ngữ."""
    try:
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
    
    # Kiểm tra pattern lặp còn sót
    is_repeated, cleaned = detect_repeated_pattern(text)
    if is_repeated:
        print(f"   🔄 Found repeated pattern in validation: '{text}' -> using '{cleaned}'")
        text = cleaned
    
    # Chỉ cảnh báo nếu quá ngắn
    words = text.split()
    if len(words) < 3:
        print(f"   ⚠️ Warning: Very short text ({len(words)} words), this may cause issues")
    
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
        
        for i, (sentence, pause_duration, is_merged) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:80]}...")
            
            # Chuẩn hóa văn bản an toàn
            normalized_text = post_process(safe_normalize(sentence))
            
            # Validate văn bản
            normalized_text = validate_text_for_tts(normalized_text)
            
            # Kiểm tra độ dài tối thiểu
            word_count = len(normalized_text.strip().split())
            if word_count < 2:
                print(f"   ⏭️ Skipped (too short: {word_count} words): '{normalized_text}'")
                continue
            
            print(f"   📝 Normalized ({word_count} words): {normalized_text[:80]}...")
            if is_merged:
                print(f"   ℹ️ Merged sentence - model will create natural pauses")
            
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
                    
                    # Thêm khoảng im lặng giữa các chunk
                    if i < len(chunks) - 1 and not is_merged:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Added {pause_duration}s silence")
                    elif i < len(chunks) - 1 and is_merged:
                        print(f"   🔇 No manual silence (merged sentence)")
                        
                except Exception as e:
                    retry_count += 1
                    print(f"   ⚠️ Attempt {retry_count} failed: {str(e)[:100]}")
                    
                    if retry_count > max_retries:
                        print(f"   ❌ Max retries reached, skipping chunk")
                        # Thử với văn bản đơn giản hơn
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
                            except:
                                print(f"   ❌ Simplified attempt also failed, skipping")
                        break
                    
                    # Đợi một chút trước khi retry
                    import time
                    time.sleep(0.5)
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text or try simpler sentences.")
            
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
    
    ✨ **New**: Automatically handles repeated words like "Há há há", "A a a"
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Test repeated words:

Chớp mắt một cái bỗng dưng không còn nhìn thấy bé Tư đâu nữa. Trong bóng đêm dày đặc chỉ nghe thấy tiếng cười quỷ dị của y. 

"Há há há..."

Minh Huy căng mắt nhìn ra xung quanh. Mồ hôi trên trán rơi xuống mi mắt hắn một mảng cay xè. 

"A!!!!!!!" """, 
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
    ### 🆕 Repeated Word Handling:
    
    | Input | Output | Action |
    |-------|--------|--------|
    | `"Há há há..."` | `"Há"` merged with adjacent sentence | ✅ Fixed |
    | `"A a a!!!"` | `"A"` merged with next sentence | ✅ Fixed |
    | `"He he he"` | `"He"` merged with previous sentence | ✅ Fixed |
    
    ### 💡 How It Works:
    1. **Detects** repeated words (same word 2+ times)
    2. **Simplifies** to single occurrence
    3. **Merges** with adjacent sentence
    4. **Skips** if merging fails (prevents crash)
    
    ### 📖 Features:
    - ✅ Handles repeated laughter ("Há há há")
    - ✅ Handles repeated exclamations ("A a a!!!")
    - ✅ Cleans excessive punctuation ("!!!!" → "!")
    - ✅ Smart merging with context
    - ✅ Graceful skipping if unprocessable
    
    ### ⚠️ Note:
    - Repeated words are simplified to avoid TTS model issues
    - If a repeated pattern can't be merged, it will be skipped
    - Check console logs for processing details
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Repeated Words**: Now handled automatically (merged or skipped)
        4. **Processing Time**: Increases with text length
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
