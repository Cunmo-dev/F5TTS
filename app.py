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
    Tách văn bản theo chuẩn repo gốc:
    - Merge câu < 4 từ bằng dấu PHẨY (không phải chấm)
    - Merge với câu trước nếu có, nếu không thì câu sau
    """
    chunks = []
    
    # Tách theo dòng trống
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Kiểm tra hội thoại
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
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
                if sentence_text:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
        
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    # ===== LOGIC MERGE THEO REPO GỐC =====
    # Merge câu < 4 từ bằng DẤU PHẨY
    i = 0
    while i < len(chunks):
        sentence, pause = chunks[i]
        word_count = len(sentence.split())
        
        if word_count < 4:
            if i == 0 and len(chunks) > 1:
                # Câu đầu tiên: merge với câu SAU bằng phẩy
                next_sentence, next_pause = chunks[i + 1]
                merged = sentence + ', ' + next_sentence
                chunks[i] = (merged, next_pause)
                del chunks[i + 1]
                print(f"   🔗 Merged first short sentence: '{sentence}' + '{next_sentence[:30]}...'")
            elif i > 0:
                # Câu giữa/cuối: merge với câu TRƯỚC bằng phẩy
                prev_sentence, prev_pause = chunks[i - 1]
                merged = prev_sentence + ', ' + sentence
                chunks[i - 1] = (merged, prev_pause)
                del chunks[i]
                i -= 1  # Lùi lại để kiểm tra câu tiếp theo
                print(f"   🔗 Merged short sentence with previous: '{sentence}'")
            else:
                # Trường hợp đặc biệt: chỉ có 1 câu ngắn
                # Lặp lại từ cuối để đủ 4 từ
                words = sentence.split()
                while len(words) < 4:
                    # Lấy từ cuối (bỏ dấu câu nếu có)
                    last_word = re.sub(r'[.,!?;:\-…]+$', '', words[-1])
                    words.append(last_word)
                padded = ' '.join(words)
                chunks[i] = (padded, pause)
                print(f"   ⚠️ Padded single short sentence: '{sentence}' → '{padded}'")
        
        i += 1
    
    return chunks

def create_silence(duration_seconds, sample_rate=24000):
    """Tạo đoạn im lặng."""
    num_samples = int(duration_seconds * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

def post_process(text):
    """Làm sạch văn bản - loại bỏ tất cả dấu câu dư thừa."""
    text = " " + text + " "
    
    # Loại bỏ dấu ngoặc kép
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    
    # Loại bỏ dấu chấm/phẩy/chấm than/chấm hỏi dư thừa
    text = re.sub(r'\.{2,}', '.', text)  # Nhiều dấu chấm → 1 dấu chấm
    text = re.sub(r',+', ',', text)      # Nhiều dấu phẩy → 1 dấu phẩy
    text = re.sub(r'!+', '!', text)      # Nhiều dấu chấm than → 1
    text = re.sub(r'\?+', '?', text)     # Nhiều dấu hỏi → 1
    
    # Loại bỏ dấu câu ở cuối (TTS không cần)
    text = re.sub(r'[.,!?;:\-…]+

def safe_normalize(text):
    """Normalize văn bản an toàn - BỎ QUA các câu lặp từ đơn giản."""
    # Kiểm tra xem có phải câu lặp từ đơn giản không (há há, hê hê, à à...)
    words = text.lower().strip().split()
    unique_words = set(re.sub(r'[.,!?;:\-…]+', '', w) for w in words)
    
    # Nếu chỉ có 1-2 từ duy nhất được lặp lại → KHÔNG normalize
    if len(unique_words) <= 2 and len(words) <= 5:
        cleaned = re.sub(r'[.,!?;:\-…]+', '', text.lower().strip())
        print(f"   🎭 Detected repetitive pattern, skipped normalize: '{cleaned}'")
        return cleaned
    
    # Các câu bình thường → normalize
    try:
        normalized = TTSnorm(text)
        if len(normalized.strip()) < 2:
            return text.lower()
        return normalized.lower()
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original text")
        return text.lower()

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
    """TTS inference với pause thực sự."""
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        pause_configs = {
            "Short": (0.2, 0.1),
            "Medium": (0.4, 0.2),
            "Long": (0.6, 0.3)
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.4, 0.2))
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks after merge: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks, 1):
            word_count = len(sent.split())
            print(f"   {idx}. [{word_count} words, {pause}s] {sent[:100]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Generate audio
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing...")
            print(f"   📄 Original: {sentence[:100]}")
            
            # Normalize
            normalized_text = post_process(safe_normalize(sentence))
            normalized_text = ' '.join(normalized_text.split())  # Clean whitespace
            
            word_count = len(normalized_text.strip().split())
            print(f"   📝 Normalized ({word_count} words): {normalized_text[:100]}")
            
            # QUAN TRỌNG: Nếu < 3 từ → pad thêm
            if word_count < 3:
                # Lặp lại toàn bộ text
                original_words = normalized_text.split()
                while len(original_words) < 3:
                    original_words.extend(normalized_text.split())
                normalized_text = ' '.join(original_words[:5])  # Tối đa 5 từ
                print(f"   ➕ Padded to {len(normalized_text.split())} words: {normalized_text}")
            
            # Kiểm tra lại sau khi pad
            final_word_count = len(normalized_text.strip().split())
            if final_word_count < 2:
                print(f"   ⏭️ Still too short after padding, skipping")
                continue
            
            # Retry với backoff
            max_retries = 3
            success = False
            
            for retry in range(max_retries + 1):
                try:
                    if retry > 0:
                        print(f"   🔁 Retry {retry}/{max_retries}...")
                        # Thử làm sạch hơn nữa
                        normalized_text = re.sub(r'[^\w\s]', '', normalized_text)
                        normalized_text = ' '.join(normalized_text.split())
                        print(f"   🧹 Extra cleaned: {normalized_text}")
                    
                    print(f"   🎤 Calling TTS with: ref_text='{ref_text[:30]}', gen_text='{normalized_text[:50]}'")
                    
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
                    
                    # Add silence between chunks
                    if i < len(chunks) - 1:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Added {pause_duration}s silence")
                    
                    break
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"   ❌ Attempt {retry + 1} failed: {error_msg[:100]}")
                    
                    if retry == max_retries:
                        print(f"   ⚠️ Max retries reached, skipping this chunk")
                        print(f"   📊 Debug info: text_length={len(normalized_text)}, word_count={len(normalized_text.split())}")
                    else:
                        import time
                        time.sleep(0.5)
        
        if not audio_segments:
            raise gr.Error("No audio generated. Please check your text.")
        
        # Concat all audio
        final_wave = np.concatenate(audio_segments)
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s from {len(chunks)} chunks")
        
        # Create spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spec:
            spectrogram_path = tmp_spec.name
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
        raise gr.Error(f"Error: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech (Fixed - Repo Standard)
    
    ✨ **Following original repo logic**: Short sentences (< 4 words) merged with COMMA
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Example:
Chớp mắt một cái bỗng dưng không còn nhìn thấy bé Tư đâu nữa. Trong bóng đêm dày đặc chỉ nghe thấy tiếng cười quỷ dị của y.

"Há há há..."

Minh Huy căng mắt nhìn ra xung quanh. Mồ hôi trên trán rơi xuống mi mắt hắn một mảng cay xè.

"A!!!!!!!"
""", 
            lines=10
        )
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, 1.0, 0.1, label="⚡ Speed")
        pause_level = gr.Radio(
            ["Short", "Medium", "Long"],
            value="Medium",
            label="⏸️ Pause Duration"
        )
    
    btn = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spec = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 Logic theo repo gốc:
    
    - **Câu < 4 từ**: Merge với câu trước/sau bằng **dấu phẩy** (`,`)
    - **Câu đầu tiên ngắn**: Merge với câu sau
    - **Câu giữa/cuối ngắn**: Merge với câu trước
    - **Chỉ 1 câu ngắn**: Lặp lại từ cuối để đủ 4 từ
    
    ### 📖 Ví dụ xử lý:
    ```
    Input:
    "Tiếng cười của y.
    Há há há...
    Minh Huy căng mắt."
    
    → "Há há há..." chỉ có 1 từ (< 4)
    → Merge với câu trước: "Tiếng cười của y, há há há..."
    → Câu này giờ có 7 từ → OK ✅
    ```
    """)

    btn.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spec]
    )

demo.queue().launch(share=True)
, '', text.strip())
    
    return " ".join(text.split())

def safe_normalize(text):
    """Normalize văn bản an toàn."""
    try:
        normalized = TTSnorm(text)
        if len(normalized.strip()) < 2:
            return text.lower()
        return normalized.lower()
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original text")
        return text.lower()

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
    """TTS inference với pause thực sự."""
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        pause_configs = {
            "Short": (0.2, 0.1),
            "Medium": (0.4, 0.2),
            "Long": (0.6, 0.3)
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.4, 0.2))
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks after merge: {len(chunks)}")
        for idx, (sent, pause) in enumerate(chunks, 1):
            word_count = len(sent.split())
            print(f"   {idx}. [{word_count} words, {pause}s] {sent[:100]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Generate audio
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing...")
            
            # Normalize
            normalized_text = post_process(safe_normalize(sentence))
            normalized_text = ' '.join(normalized_text.split())  # Clean whitespace
            
            word_count = len(normalized_text.strip().split())
            print(f"   📝 Text ({word_count} words): {normalized_text[:100]}...")
            
            # Kiểm tra độ dài tối thiểu
            if word_count < 3:
                print(f"   ⚠️ Text too short after normalization, skipping")
                continue
            
            # Retry với backoff
            max_retries = 3
            success = False
            
            for retry in range(max_retries + 1):
                try:
                    if retry > 0:
                        print(f"   🔁 Retry {retry}/{max_retries}...")
                    
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
                    
                    # Add silence between chunks
                    if i < len(chunks) - 1:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Added {pause_duration}s silence")
                    
                    break
                    
                except Exception as e:
                    print(f"   ⚠️ Attempt {retry + 1} failed: {str(e)[:80]}")
                    if retry == max_retries:
                        print(f"   ❌ Max retries reached, skipping chunk")
                    else:
                        import time
                        time.sleep(0.5)
        
        if not audio_segments:
            raise gr.Error("No audio generated. Please check your text.")
        
        # Concat all audio
        final_wave = np.concatenate(audio_segments)
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s from {len(chunks)} chunks")
        
        # Create spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spec:
            spectrogram_path = tmp_spec.name
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
        raise gr.Error(f"Error: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech (Fixed - Repo Standard)
    
    ✨ **Following original repo logic**: Short sentences (< 4 words) merged with COMMA
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Example:
Chớp mắt một cái bỗng dưng không còn nhìn thấy bé Tư đâu nữa. Trong bóng đêm dày đặc chỉ nghe thấy tiếng cười quỷ dị của y.

"Há há há..."

Minh Huy căng mắt nhìn ra xung quanh. Mồ hôi trên trán rơi xuống mi mắt hắn một mảng cay xè.

"A!!!!!!!"
""", 
            lines=10
        )
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, 1.0, 0.1, label="⚡ Speed")
        pause_level = gr.Radio(
            ["Short", "Medium", "Long"],
            value="Medium",
            label="⏸️ Pause Duration"
        )
    
    btn = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spec = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 Logic theo repo gốc:
    
    - **Câu < 4 từ**: Merge với câu trước/sau bằng **dấu phẩy** (`,`)
    - **Câu đầu tiên ngắn**: Merge với câu sau
    - **Câu giữa/cuối ngắn**: Merge với câu trước
    - **Chỉ 1 câu ngắn**: Lặp lại từ cuối để đủ 4 từ
    
    ### 📖 Ví dụ xử lý:
    ```
    Input:
    "Tiếng cười của y.
    Há há há...
    Minh Huy căng mắt."
    
    → "Há há há..." chỉ có 1 từ (< 4)
    → Merge với câu trước: "Tiếng cười của y, há há há..."
    → Câu này giờ có 7 từ → OK ✅
    ```
    """)

    btn.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spec]
    )

demo.queue().launch(share=True)
