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

def is_repetitive_text(text):
    """
    Kiểm tra xem văn bản có phải là lặp lại không (há há há, hahaha, A!!!!, etc.)
    """
    # Loại bỏ khoảng trắng và ký tự đặc biệt để kiểm tra
    clean = re.sub(r'[^\w]', '', text.lower())
    
    # Nếu quá ngắn, không coi là lặp
    if len(clean) < 2:
        return False
    
    # Kiểm tra pattern lặp: "hahaha", "hihihi"
    if len(set(clean)) <= 2 and len(clean) >= 4:
        return True
    
    # Kiểm tra từ lặp với khoảng trắng: "há há há"
    words = text.lower().split()
    if len(words) >= 2:
        # Nếu tất cả các từ giống nhau
        if len(set(words)) == 1:
            return True
        # Nếu có ít nhất 3 từ và 80% giống nhau
        if len(words) >= 3:
            most_common = max(set(words), key=words.count)
            if words.count(most_common) / len(words) >= 0.8:
                return True
    
    return False

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu, chỉ ghép câu < 3 từ hoặc câu lặp lại bằng dấu chấm.
    
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
    
    # Gộp các câu < 3 từ HOẶC câu lặp lại bằng dấu chấm
    # NHƯNG GIỮ NGUYÊN PAUSE CỦA TỪNG CÂU RIÊNG LẺ
    merged_chunks = []
    temp_sentences = []  # Danh sách các câu tích lũy
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, _) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        is_repetitive = is_repetitive_text(sentence)
        
        # Kiểm tra điều kiện ghép: < 3 từ HOẶC là câu lặp
        should_merge = word_count < 3 or is_repetitive
        
        if not should_merge:
            # Câu đủ dài và không lặp
            if temp_sentences:
                # Gộp các câu tích lũy + câu hiện tại bằng dấu chấm
                all_sentences = temp_sentences + [sentence]
                merged_text = ". ".join(all_sentences)
                # FIX: Vì đã gộp câu bằng dấu chấm, model sẽ tự tạo pause
                # Chỉ cần pause NGẮN sau chunk này (không cần pause dài)
                merged_chunks.append((merged_text, pause, True))
                temp_sentences = []
                if is_repetitive:
                    print(f"   🔗 Merged repetitive text: '{sentence[:50]}...'")
            else:
                # Câu độc lập - giữ nguyên pause
                merged_chunks.append((sentence, pause, False))
        else:
            # Câu ngắn (< 3 từ) HOẶC câu lặp, tích lũy
            temp_sentences.append(sentence)
            temp_pause = pause
            
            if is_repetitive:
                print(f"   ⚠️ Detected repetitive text: '{sentence}', will merge with adjacent sentence")
            
            # Nếu là câu cuối -> gộp với câu trước
            if is_last:
                if merged_chunks:
                    # Gộp vào câu trước bằng dấu chấm
                    last_sentence, last_pause, last_merged = merged_chunks[-1]
                    combined_text = last_sentence + ". " + ". ".join(temp_sentences)
                    # FIX: Giữ nguyên pause của câu trước (đã gộp rồi không cần pause dài)
                    merged_chunks[-1] = (combined_text, last_pause, True)
                    print(f"   🔗 Merged last short/repetitive chunk(s) with period")
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
            # FIX: Giữ nguyên pause của câu trước
            merged_chunks[-1] = (combined_text, last_pause, True)
            print(f"   🔗 Merged remaining short/repetitive chunks with period")
        else:
            # Trường hợp đặc biệt: chỉ có câu ngắn
            merged_text = ". ".join(temp_sentences)
            while len(merged_text.split()) < 3:
                merged_text += " này"
            print(f"   ⚠️ Only short/repetitive sentence(s) found, padded: '{merged_text}'")
            merged_chunks.append((merged_text, temp_pause, False))
    
    return merged_chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng với thời gian xác định."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Làm sạch văn bản, loại bỏ ký tự đặc biệt (giữ lại dấu chấm)."""
    text = " " + text + " "
    
    # Loại bỏ dấu chấm lặp
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = re.sub(r'\.{2,}', '.', text)  # Thay ... thành .
    
    # Loại bỏ dấu ngoặc kép
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    
    # Loại bỏ dấu phẩy dư thừa
    text = re.sub(r',+', ',', text)
    
    # Loại bỏ các ký tự đặc biệt NGOẠI TRỪ dấu chấm và chữ cái/số
    # Giữ lại: chữ cái, số, khoảng trắng, dấu chấm
    text = re.sub(r'[^\w\s.]', '', text, flags=re.UNICODE)
    
    # Loại bỏ khoảng trắng thừa
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
                    
                    # LOGIC TỐI ƯU: 
                    # - Nếu là câu MERGED: Model đã tự tạo pause tại dấu chấm -> CHỈ THÊM SILENCE NGẮN
                    # - Nếu là câu SINGLE: Cần thêm silence đầy đủ
                    if i < len(chunks) - 1:
                        if is_merged:
                            # Câu gộp: Model đã pause tại dấu chấm -> chỉ cần thêm 50% silence
                            reduced_pause = pause_duration * 0.5
                            silence = create_silence(reduced_pause, sample_rate)
                            audio_segments.append(silence)
                            print(f"   ⏸️  Added {reduced_pause}s reduced silence (merged sentence)")
                        else:
                            # Câu đơn: Thêm silence đầy đủ như code gốc
                            silence = create_silence(pause_duration, sample_rate)
                            audio_segments.append(silence)
                            print(f"   ⏸️  Added {pause_duration}s full silence")
                        
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
    ### 💡 How Smart Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Paragraph Detection** | Separates narrative text by double line breaks |
    | **Dialogue Detection** | Identifies quoted speech (even multi-line) |
    | **Smart Period Merging** | Sentences < 3 words OR repetitive text are merged with periods |
    | **Repetitive Text Handling** | Auto-detects "há há há", "hahaha", etc. and merges with adjacent sentences |
    | **Model-Based Pauses** | AI naturally pauses at periods |
    | **Special Character Removal** | Removes !?... etc., keeps only letters, numbers, spaces and periods |
    | **Consistent Silence** | ALWAYS adds silence between chunks (like original code) |
    | **Three Levels** | Short (0.2s/0.1s), Medium (0.4s/0.2s), Long (0.6s/0.3s) |
    
    ### 📖 Usage Tips:
    - **Separate paragraphs** with double line breaks (`\n\n`)
    - **Dialogue** can span multiple lines - just use quotes `"..."`
    - **Short sentences** (< 3 words) are automatically merged
    - **Repetitive sounds** like "Há há há..." are merged with nearby sentences
    - **Special characters** (!!!, ???, ...) are automatically removed
    - **Natural prosody**: Model creates pauses at periods + manual silence between chunks
    - **Short**: Fast-paced reading
    - **Medium**: Natural storytelling (recommended)
    - **Long**: Dramatic audiobooks
    
    ### 🎯 Example Processing:
    ```
    Input:
    "Há há há..."
    Minh Huy căng mắt nhìn.
    
    → "Há há há..." is repetitive, merged as:
    "Há há há. Minh Huy căng mắt nhìn."
    + 0.4s silence after this chunk
    
    Input:
    "A!!!!!!!"
    
    → Special characters removed:
    "A" → Too short, merged with next sentence
    
    Input:
    Chớp mắt một cái. "Há há há..." Minh Huy căng mắt.
    
    → Processed as:
    "Chớp mắt một cái. Há há há. Minh Huy căng mắt."
    + silence after each chunk
    ```
    
    ### ⚠️ Note:
    - Each sentence is processed separately, then combined with real silence
    - Sentences with < 3 words OR repetitive patterns are merged using periods
    - All special characters except periods are removed before TTS
    - **Silence is ALWAYS added between chunks** (merged or not)
    - Longer texts take more time but produce better pause quality
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
        5. **Foreign Words**: Pronounced phonetically in Vietnamese
        6. **Very Short Sentences**: Sentences < 3 words are automatically merged
        7. **Repetitive Text**: Patterns like "há há há" are merged with adjacent sentences
        8. **Special Characters**: All special characters except periods are removed (!!!, ???, ... → removed)
        9. **Error Recovery**: If one sentence fails, processing continues with remaining text
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
