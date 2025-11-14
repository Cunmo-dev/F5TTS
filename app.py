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
    Kiểm tra xem câu có phải là văn bản lặp lại không.
    Ví dụ: "Há há há", "hu hu hu", "ha ha ha..."
    
    Returns:
        bool: True nếu là văn bản lặp lại
    """
    # Loại bỏ dấu câu và chuyển về lowercase
    clean_text = re.sub(r'[.!?,;:]', '', text).lower().strip()
    
    # Tách thành các từ
    words = clean_text.split()
    
    # Nếu ít hơn 2 từ, không coi là lặp
    if len(words) < 2:
        return False
    
    # Kiểm tra xem tất cả các từ có giống nhau không
    unique_words = set(words)
    
    # Nếu chỉ có 1 từ duy nhất được lặp lại
    if len(unique_words) == 1:
        return True
    
    # Kiểm tra pattern lặp với variation nhỏ (ví dụ: "hà hà", "há há")
    # Loại bỏ dấu thanh để so sánh
    normalized_words = []
    for word in words:
        # Loại bỏ các ký tự đặc biệt và dấu thanh (giữ lại chữ cái cơ bản)
        base_word = ''.join(c for c in word if c.isalpha())
        normalized_words.append(base_word)
    
    unique_normalized = set(normalized_words)
    
    # Nếu sau khi normalize chỉ còn 1 từ -> là lặp
    if len(unique_normalized) == 1 and len(normalized_words) >= 2:
        return True
    
    # Kiểm tra pattern lặp (ví dụ: "ha ha", "he he")
    # Nếu 80% từ giống nhau (cho phép một vài variation)
    if len(unique_normalized) <= max(2, len(words) * 0.3):
        return True
    
    return False

def clean_text_before_processing(text):
    """
    Làm sạch text trước khi xử lý: loại bỏ emoji và ký tự đặc biệt.
    """
    # Loại bỏ emoji (Unicode ranges)
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"  # supplemental symbols
        "\U0001FA00-\U0001FA6F"
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    
    # Loại bỏ khoảng trắng thừa
    text = ' '.join(text.split())
    
    return text.strip()

def extract_quoted_segments(text):
    """
    Trích xuất các đoạn trong ngoặc kép và text bên ngoài.
    
    Returns:
        list of tuples: [(text, is_quoted), ...]
    """
    # Làm sạch text trước
    text = clean_text_before_processing(text)
    
    segments = []
    # Pattern để tìm text trong ngoặc kép (hỗ trợ cả ", ", ")
    pattern = r'(["""])([^"""]+)(["""])'
    
    last_end = 0
    for match in re.finditer(pattern, text):
        # Text trước ngoặc kép
        before_text = text[last_end:match.start()].strip()
        if before_text:
            segments.append((before_text, False))
        
        # Text trong ngoặc kép (không bao gồm dấu ngoặc)
        quoted_text = match.group(2).strip()
        if quoted_text:
            segments.append((quoted_text, True))
        
        last_end = match.end()
    
    # Text sau ngoặc kép cuối cùng
    after_text = text[last_end:].strip()
    if after_text:
        segments.append((after_text, False))
    
    return segments

def split_text_into_sentences(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu, giữ nguyên câu trong ngoặc kép.
    
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
        
        # Tách các dòng trong đoạn
        lines = para.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Trích xuất các segment (quoted và non-quoted)
            segments = extract_quoted_segments(line)
            
            if not segments:
                continue
            
            for segment_text, is_quoted in segments:
                if not segment_text.strip():
                    continue
                
                if is_quoted:
                    # Câu trong ngoặc kép -> giữ nguyên, dùng pause dialogue
                    chunks.append((segment_text, pause_dialogue_duration, False))
                    print(f"   💬 Quoted dialogue: '{segment_text[:60]}...'")
                else:
                    # Text bên ngoài ngoặc kép -> tách như bình thường
                    sentences = re.split(r'([.!?]+)', segment_text)
                    
                    current_sentence = ""
                    for i, part in enumerate(sentences):
                        if i % 2 == 0:  # Phần văn bản
                            current_sentence += part
                        else:  # Dấu câu
                            current_sentence += part
                            sentence_text = current_sentence.strip()
                            
                            if sentence_text:
                                chunks.append((sentence_text, pause_paragraph_duration, False))
                                current_sentence = ""
                    
                    # Thêm phần còn lại nếu có
                    if current_sentence.strip():
                        chunks.append((current_sentence.strip(), pause_paragraph_duration, False))
    
    # Gộp các câu < 3 từ hoặc câu lặp lại bằng dấu chấm
    merged_chunks = []
    temp_sentences = []
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, _) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        is_repetitive = is_repetitive_text(sentence)
        
        # Kiểm tra xem câu có cần gộp không
        should_merge = (word_count < 3) or is_repetitive
        
        if not should_merge:
            # Câu đủ dài và không lặp
            if temp_sentences:
                # Gộp các câu tích lũy + câu hiện tại bằng dấu chấm
                all_sentences = temp_sentences + [sentence]
                merged_text = ". ".join(all_sentences)
                merged_chunks.append((merged_text, pause, True))
                print(f"   🔗 Merged sentences: '{merged_text[:80]}...'")
                temp_sentences = []
            else:
                # Câu độc lập
                merged_chunks.append((sentence, pause, False))
        else:
            # Câu ngắn hoặc lặp, tích lũy
            if is_repetitive:
                print(f"   🔁 Detected repetitive text: '{sentence}' - will merge")
            temp_sentences.append(sentence)
            temp_pause = pause
            
            # Nếu là câu cuối -> gộp với câu trước
            if is_last:
                if merged_chunks:
                    last_sentence, last_pause, last_merged = merged_chunks[-1]
                    combined_text = last_sentence + ". " + ". ".join(temp_sentences)
                    merged_chunks[-1] = (combined_text, last_pause, True)
                    print(f"   🔗 Merged last short/repetitive chunk(s)")
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
            print(f"   🔗 Merged remaining short/repetitive chunks")
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
    """Làm sạch văn bản - loại bỏ tất cả ký tự đặc biệt."""
    # Loại bỏ tất cả dấu câu và ký tự đặc biệt, chỉ giữ chữ cái, số và khoảng trắng
    # Giữ lại các ký tự tiếng Việt có dấu
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split()).strip()
    
    return text

def safe_normalize(text):
    """Normalize văn bản an toàn, xử lý lỗi với từ ngoại ngữ."""
    try:
        normalized = TTSnorm(text)
        if len(normalized.strip()) < 2:
            return text.lower()
        return normalized.lower()
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original text")
        return text.lower()

def validate_text_for_tts(text):
    """Kiểm tra văn bản trước khi đưa vào TTS."""
    text = ' '.join(text.split())
    
    words = text.split()
    if len(words) < 3:
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
              pause_paragraph: float = 0.4, pause_dialogue: float = 0.2, request: gr.Request = None):
    """
    TTS inference với pause thực sự bằng cách ghép audio.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản thành các câu với thời gian dừng
        chunks = split_text_into_sentences(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        for idx, (sent, pause, is_merged) in enumerate(chunks[:5], 1):
            marker = "🔗 MERGED" if is_merged else "📄 SINGLE"
            print(f"   {idx}. [{marker}, {pause}s] {sent[:80]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration, is_merged) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:80]}...")
            
            # Chuẩn hóa văn bản
            normalized_text = safe_normalize(sentence)
            normalized_text = post_process(normalized_text)
            normalized_text = validate_text_for_tts(normalized_text)
            normalized_text = normalized_text.rstrip('.')
            
            word_count = len(normalized_text.strip().split())
            if word_count < 1:
                print(f"   ⏭️ Skipped (empty)")
                continue
            
            if word_count < 3:
                original_text = normalized_text
                normalized_text = normalized_text + " này"
                print(f"   ⚠️ Short sentence padded: '{original_text}' -> '{normalized_text}'")
            
            print(f"   📝 Normalized ({len(normalized_text.split())} words): {normalized_text[:80]}...")
            
            # Retry logic
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
                    
                    # Thêm im lặng giữa các chunk
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
                        print(f"   ❌ Max retries reached, skipping")
                        if len(normalized_text.split()) > 3:
                            print(f"   🔧 Trying with first 3 words...")
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
                                print(f"   ❌ Simplified attempt failed")
                        break
                    
                    import time
                    time.sleep(0.5)
        
        # Ghép tất cả audio
        if not audio_segments:
            raise gr.Error("No valid audio segments generated.")
            
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
    gr.Markdown("# 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis")
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="Enter Vietnamese text here...", 
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
        pause_paragraph = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.4,
            step=0.05,
            label="⏸️ Pause (Paragraph)"
        )
        pause_dialogue = gr.Slider(
            minimum=0.0,
            maximum=1.0,
            value=0.2,
            step=0.05,
            label="⏸️ Pause (Dialogue)"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_paragraph, pause_dialogue], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
