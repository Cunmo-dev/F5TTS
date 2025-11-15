import os
os.environ["MPLBACKEND"] = "Agg"

import spaces
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm
import re

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
    """
    clean_text = re.sub(r'[.!?,;:]', '', text).lower().strip()
    words = clean_text.split()
    
    if len(words) < 2:
        return False
    
    unique_words = set(words)
    if len(unique_words) == 1:
        return True
    
    # Loại bỏ dấu thanh để so sánh
    normalized_words = [''.join(c for c in word if c.isalpha()) for word in words]
    unique_normalized = set(normalized_words)
    
    if len(unique_normalized) == 1 and len(normalized_words) >= 2:
        return True
    
    if len(unique_normalized) <= max(2, len(words) * 0.3):
        return True
    
    return False

def normalize_sentence_ending(sentence):
    """
    Chuẩn hóa ký tự kết thúc câu:
    - Nếu không có dấu chấm câu → thêm dấu chấm
    - Nếu có dấu chấm + ký tự đặc biệt → xóa ký tự đặc biệt
    """
    sentence = sentence.strip()
    
    # Danh sách dấu câu hợp lệ
    valid_punctuation = '.!?'
    
    # Kiểm tra ký tự cuối
    if not sentence:
        return sentence + "."
    
    last_char = sentence[-1]
    
    # Nếu đã có dấu câu hợp lệ
    if last_char in valid_punctuation:
        # Xóa các ký tự đặc biệt sau dấu chấm (nếu có)
        while len(sentence) > 1 and sentence[-1] not in valid_punctuation:
            sentence = sentence[:-1]
        return sentence
    
    # Kiểm tra có dấu câu ở vị trí gần cuối không
    for i in range(len(sentence) - 1, max(0, len(sentence) - 5), -1):
        if sentence[i] in valid_punctuation:
            # Có dấu câu nhưng có ký tự đặc biệt phía sau → cắt bỏ
            return sentence[:i+1]
    
    # Không có dấu câu → thêm dấu chấm
    return sentence + "."

def smart_text_preprocessing(text, silence_duration=0.4):
    """
    Xử lý văn bản thông minh:
    - Phát hiện và xử lý câu lặp lại (há há há)
    - Gộp câu ngắn < 3 từ bằng dấu chấm
    - Chuẩn hóa ký tự kết thúc câu
    - Thêm dấu chấm lặp để tạo pause tự nhiên (dựa vào silence_duration)
    
    Returns:
        str: Văn bản đã được xử lý, sẵn sàng đọc một lần
    """
    print("\n📝 Starting smart text preprocessing...")
    print(f"   Silence duration: {silence_duration}s (will add extra periods)")
    
    # Tính số dấu chấm cần thêm dựa vào silence duration
    # 0.1-0.3s: 1 dấu chấm
    # 0.4-0.6s: 2 dấu chấm
    # 0.7-1.0s: 3 dấu chấm
    if silence_duration <= 0.3:
        pause_marker = "."
        para_pause_marker = ". "
    elif silence_duration <= 0.6:
        pause_marker = ". "
        para_pause_marker = ". . "
    else:
        pause_marker = ". . "
        para_pause_marker = ". . . "
    
    print(f"   Using pause marker: '{pause_marker}' between sentences")
    print(f"   Using para marker: '{para_pause_marker}' between paragraphs")
    
    # Tách theo đoạn văn
    paragraphs = text.split('\n\n')
    processed_paragraphs = []
    
    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        print(f"\n📄 Processing paragraph {para_idx + 1}:")
        
        # Loại bỏ dấu ngoặc kép để xử lý
        clean_para = para.replace('"', '').replace('"', '').replace('"', '').strip()
        
        # Tách thành các câu
        sentences = re.split(r'([.!?]+)', clean_para)
        
        processed_sentences = []
        temp_accumulator = []  # Tích lũy câu ngắn/lặp
        
        for i in range(0, len(sentences), 2):
            if i >= len(sentences):
                break
                
            sentence_text = sentences[i].strip()
            punctuation = sentences[i + 1] if i + 1 < len(sentences) else '.'
            
            if not sentence_text:
                continue
            
            full_sentence = sentence_text + punctuation
            word_count = len(sentence_text.split())
            is_repetitive = is_repetitive_text(sentence_text)
            
            print(f"   - '{sentence_text[:50]}...' ({word_count} words, repetitive: {is_repetitive})")
            
            # Kiểm tra xem có cần gộp không
            should_merge = (word_count < 3) or is_repetitive
            
            if should_merge:
                print(f"     → Will merge (too short or repetitive)")
                temp_accumulator.append(sentence_text)
            else:
                # Câu đủ dài
                if temp_accumulator:
                    # Gộp các câu tích lũy + câu hiện tại
                    merged = ". ".join(temp_accumulator + [sentence_text])
                    merged = normalize_sentence_ending(merged)
                    processed_sentences.append(merged)
                    print(f"     → Merged with accumulated: '{merged[:60]}...'")
                    temp_accumulator = []
                else:
                    # Câu độc lập
                    normalized = normalize_sentence_ending(sentence_text + punctuation)
                    processed_sentences.append(normalized)
                    print(f"     → Kept as is: '{normalized[:60]}...'")
        
        # Xử lý câu còn sót
        if temp_accumulator:
            if processed_sentences:
                # Gộp vào câu trước
                last_sentence = processed_sentences[-1].rstrip('.!?')
                merged = last_sentence + ". " + ". ".join(temp_accumulator)
                merged = normalize_sentence_ending(merged)
                processed_sentences[-1] = merged
                print(f"   🔗 Merged remaining to last sentence")
            else:
                # Chỉ có câu ngắn → giữ nguyên, KHÔNG thêm gì cả
                merged = ". ".join(temp_accumulator)
                merged = normalize_sentence_ending(merged)
                processed_sentences.append(merged)
                print(f"   ⚠️ Only short sentences (kept as is): '{merged}'")
        
        # Ghép các câu trong đoạn với pause marker (dấu chấm)
        processed_para = pause_marker.join(processed_sentences)
        processed_paragraphs.append(processed_para)
        print(f"   ✅ Paragraph result: '{processed_para[:80]}...'")
    
    # Ghép tất cả đoạn văn lại với pause dài hơn
    final_text = para_pause_marker.join(processed_paragraphs)
    
    print(f"\n✅ Preprocessing complete!")
    print(f"   Original length: {len(text)} chars")
    print(f"   Processed length: {len(final_text)} chars")
    print(f"   Preview: '{final_text[:100]}...'")
    
    return final_text

def post_process(text):
    """Làm sạch văn bản."""
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = text.replace(" .. ", " . ")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = text.replace('"', "")
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\.+', '.', text)  # Loại bỏ dấu chấm trùng
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
              silence_duration: float = 0.4, use_smart_processing: bool = True, 
              request: gr.Request = None):
    """
    TTS inference với xử lý thông minh nhưng vẫn đọc toàn bộ một lần.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Please enter text content with less than 1000 words.")
    
    try:
        # Bước 1: Smart preprocessing (nếu được bật)
        if use_smart_processing:
            processed_text = smart_text_preprocessing(gen_text, silence_duration)
        else:
            processed_text = gen_text
            print("\n📝 Smart processing disabled, using original text")
        
        # Bước 2: Normalize và clean
        print("\n🔄 Normalizing text...")
        normalized_text = safe_normalize(processed_text)
        final_text = post_process(normalized_text)
        
        word_count = len(final_text.split())
        print(f"✅ Final text ready ({word_count} words): '{final_text[:100]}...'")
        
        # Bước 3: Preprocess reference audio
        print("\n🎤 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Bước 4: Generate audio (một lần duy nhất)
        print("\n🎵 Generating audio (single pass)...")
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            final_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        duration = len(final_wave) / final_sample_rate
        print(f"✅ Audio generated successfully!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {final_sample_rate}Hz")
        
        # Bước 5: Save spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech (Hybrid Version)
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    **✨ Smart Single-Pass Processing:**
    - Detects and merges repetitive text ("Há há há" → merged with period)
    - Combines short sentences (< 3 words) automatically
    - Reads entire text in ONE go - no chunking errors!
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs and dialogue...

Example:
Hắn lúc này đang ngồi trên boong tàu. Mắt nhìn ra biển xa.

"Há há há!"
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
        silence_duration = gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.4,
            step=0.1,
            label="⏸️ Silence Duration (seconds)",
            info="Control pause length between sentences"
        )
    
    use_smart_processing = gr.Checkbox(
        value=True,
        label="🧠 Enable Smart Text Processing",
        info="Merge repetitive/short sentences before TTS"
    )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 How It Works:
    
    | Step | Description |
    |------|-------------|
    | 1️⃣ **Smart Preprocessing** | Merges short/repetitive sentences with periods |
    | 2️⃣ **Text Normalization** | Converts numbers, special chars to readable text |
    | 3️⃣ **Post Processing** | Cleans punctuation, whitespace |
    | 4️⃣ **Single-Pass TTS** | Reads entire text at once (no chunking!) |
    | 5️⃣ **Natural Pauses** | Model creates pauses at periods automatically |
    
    ### 🎯 Smart Processing Examples:
    
    **Before:**
    ```
    "Há há há!"
    "Ồ!"
    "Toa lần này trở về?"
    ```
    
    **After (with silence=0.4s):**
    ```
    "Há há há. Ồ. Toa lần này trở về?"
    ```
    → Model reads as ONE audio with natural pauses at periods
    
    **How Silence Duration Works:**
    - **0.1-0.3s**: Single period between sentences (`.`)
    - **0.4-0.6s**: Double period (`. `) for longer pause
    - **0.7-1.0s**: Triple period (`. . `) for dramatic pause
    - **Between paragraphs**: Automatically uses longer pause
    
    ### ✅ Advantages:
    - ✔️ No chunking errors
    - ✔️ Natural flow and rhythm
    - ✔️ Handles repetitive text (há há, hề hề, etc.)
    - ✔️ Merges ultra-short sentences
    - ✔️ Single GPU pass = faster + more stable
    - ✔️ Uses periods (not special markers) for pause control
    
    ### 📖 Usage Tips:
    - Separate paragraphs with double line breaks (`\\n\\n`)
    - Short sentences (< 3 words) will be merged automatically
    - Repetitive text like "Há há há" gets merged intelligently
    - Disable smart processing if you want raw text only
    - Adjust silence slider to control pause length
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Text Length**: Keep under 1000 words for best results
        5. **Foreign Words**: Pronounced phonetically in Vietnamese
        6. **Processing**: Single-pass means any error affects entire generation
        """)

    # Connect button
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, silence_duration, use_smart_processing], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
