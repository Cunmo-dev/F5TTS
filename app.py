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
    Ví dụ: "Há há há", "hu hu hu", "a a a"
    """
    clean_text = re.sub(r'[.!?,;:"\']', '', text).lower().strip()
    words = clean_text.split()
    
    if len(words) < 2:
        return False
    
    # Kiểm tra tất cả từ giống nhau
    unique_words = set(words)
    if len(unique_words) == 1:
        return True
    
    # Loại bỏ dấu thanh để so sánh
    normalized_words = [''.join(c for c in word if c.isalpha()) for word in words]
    unique_normalized = set(normalized_words)
    
    # Nếu sau khi normalize chỉ còn 1-2 từ duy nhất
    if len(unique_normalized) <= 2 and len(words) >= 3:
        return True
    
    return False

def normalize_sentence_ending(sentence):
    """
    Chuẩn hóa ký tự kết thúc câu:
    - Nếu không có dấu chấm câu → thêm dấu chấm
    - Nếu có dấu chấm + ký tự đặc biệt → xóa ký tự đặc biệt
    """
    sentence = sentence.strip()
    if not sentence:
        return "."
    
    valid_punctuation = '.!?'
    last_char = sentence[-1]
    
    # Nếu đã có dấu câu hợp lệ
    if last_char in valid_punctuation:
        return sentence
    
    # Kiểm tra có dấu câu ở gần cuối không
    for i in range(len(sentence) - 1, max(0, len(sentence) - 5), -1):
        if sentence[i] in valid_punctuation:
            # Cắt bỏ ký tự đặc biệt phía sau
            return sentence[:i+1]
    
    # Không có dấu câu → thêm dấu chấm
    return sentence + "."

def smart_merge_sentences(text):
    """
    Xử lý văn bản thông minh:
    - Merge câu < 3 từ với câu tiếp theo
    - Merge câu lặp lại (há há há, a a, etc.)
    - Giữ nguyên cấu trúc văn bản gốc
    - KHÔNG thêm bất kỳ từ nào vào văn bản
    
    Returns:
        str: Văn bản đã merge, chỉ có dấu chấm đơn
    """
    print("\n📝 Smart sentence merging...")
    
    # Loại bỏ dấu ngoặc kép
    clean_text = text.replace('"', '').replace('"', '').replace('"', '').strip()
    
    # Tách thành các câu dựa trên dấu câu
    sentences = re.split(r'([.!?]+)', clean_text)
    
    processed_sentences = []
    accumulator = []  # Tích lũy câu ngắn/lặp
    
    for i in range(0, len(sentences), 2):
        if i >= len(sentences):
            break
            
        sentence_text = sentences[i].strip()
        punctuation = sentences[i + 1] if i + 1 < len(sentences) else '.'
        
        if not sentence_text:
            continue
        
        word_count = len(sentence_text.split())
        is_repetitive = is_repetitive_text(sentence_text)
        
        print(f"   [{word_count}w, rep={is_repetitive}] '{sentence_text[:60]}...'")
        
        # Kiểm tra có cần merge không
        should_merge = (word_count < 3) or is_repetitive
        
        if should_merge:
            accumulator.append(sentence_text)
            print(f"      → Accumulating for merge")
        else:
            # Câu đủ dài và không lặp
            if accumulator:
                # Merge với các câu đã tích lũy
                merged = ". ".join(accumulator + [sentence_text])
                merged = normalize_sentence_ending(merged)
                processed_sentences.append(merged)
                print(f"      → Merged: '{merged[:60]}...'")
                accumulator = []
            else:
                # Câu độc lập
                normalized = normalize_sentence_ending(sentence_text + punctuation)
                processed_sentences.append(normalized)
                print(f"      → Keep as is")
    
    # Xử lý câu còn sót
    if accumulator:
        if processed_sentences:
            # Merge vào câu trước
            last = processed_sentences[-1].rstrip('.!?')
            merged = last + ". " + ". ".join(accumulator)
            processed_sentences[-1] = normalize_sentence_ending(merged)
            print(f"   🔗 Merged remaining to last sentence")
        else:
            # Chỉ có câu ngắn
            merged = ". ".join(accumulator)
            processed_sentences.append(normalize_sentence_ending(merged))
            print(f"   ⚠️ Only short sentences")
    
    # Ghép tất cả câu lại bằng khoảng trắng đơn
    final_text = " ".join(processed_sentences)
    
    print(f"\n✅ Merging done!")
    print(f"   Original: {len(text)} chars")
    print(f"   Processed: {len(final_text)} chars")
    print(f"   Sentences: {len(sentences)//2} → {len(processed_sentences)}")
    
    return final_text

def post_process(text):
    """Làm sạch văn bản - CHỈ xử lý lỗi format, KHÔNG xóa dấu chấm."""
    # Loại bỏ dấu ngoặc kép
    text = text.replace('"', '').replace('"', '').replace('"', '')
    
    # Loại bỏ dấu phẩy/chấm trùng lặp
    text = re.sub(r',+', ',', text)
    text = re.sub(r'\.{3,}', '.', text)  # CHỈ xóa 3 dấu chấm trở lên (...)
    
    # Chuẩn hóa khoảng trắng
    text = " ".join(text.split())
    
    return text

def safe_normalize(text):
    """Normalize văn bản an toàn với TTSnorm."""
    try:
        normalized = TTSnorm(text)
        if len(normalized.strip()) < 2:
            return text.lower()
        return normalized.lower()
    except Exception as e:
        print(f"   ⚠️ TTSnorm error: {e}, using original")
        return text.lower()

# Load models
print("🔄 Loading models...")
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://thanhcong190693/F5TTSVN/model_last.pt")),
    vocab_file=str(cached_path("hf://thanhcong190693/F5TTSVN/config.json")),
)
print("✅ Models loaded!")

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, 
              use_smart_merge: bool = True, request: gr.Request = None):
    """
    TTS inference - Đọc toàn bộ text một lần duy nhất.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    if len(gen_text.split()) > 1000:
        raise gr.Error("Please enter text content with less than 1000 words.")
    
    try:
        # Bước 1: Smart merge (nếu được bật)
        if use_smart_merge:
            print("\n" + "="*60)
            print("🧠 SMART MERGE ENABLED")
            print("="*60)
            processed_text = smart_merge_sentences(gen_text)
        else:
            print("\n" + "="*60)
            print("📝 SMART MERGE DISABLED - Using original text")
            print("="*60)
            processed_text = gen_text
        
        # Bước 2: Post process (làm sạch)
        print("\n🧹 Post processing...")
        cleaned_text = post_process(processed_text)
        print(f"   After cleaning: '{cleaned_text[:100]}...'")
        
        # Bước 3: Normalize với TTSnorm
        print("\n🔄 Normalizing with TTSnorm...")
        normalized_text = safe_normalize(cleaned_text)
        print(f"   After TTSnorm: '{normalized_text[:100]}...'")
        
        # Bước 4: Kiểm tra text cuối
        final_word_count = len(normalized_text.split())
        print(f"\n📊 Final text stats:")
        print(f"   Words: {final_word_count}")
        print(f"   Chars: {len(normalized_text)}")
        print(f"   Preview: '{normalized_text[:150]}...'")
        
        # Bước 5: Preprocess reference audio
        print("\n🎤 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Ref text: '{ref_text[:100]}...'")
        
        # Bước 6: Generate audio (MỘT LẦN DUY NHẤT)
        print("\n🎵 Generating audio (SINGLE PASS)...")
        print("="*60)
        
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text,  # ĐÂY LÀ TEXT HOÀN CHỈNH, ĐỌC MỘT LẦN
            model, 
            vocoder, 
            speed=speed
        )
        
        duration = len(final_wave) / final_sample_rate
        print("="*60)
        print(f"✅ Audio generated successfully!")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Sample rate: {final_sample_rate}Hz")
        print(f"   Array shape: {final_wave.shape}")
        print("="*60)
        
        # Bước 7: Save spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        print("\n" + "="*60)
        print("❌ ERROR OCCURRED:")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        raise gr.Error(f"Error generating voice: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS Vietnamese - Simple Single Pass
    ### ✨ Clean & Reliable Text-to-Speech
    
    **Key Features:**
    - 🎯 Reads entire text in ONE pass (no chunking errors)
    - 🧠 Smart merge: Combines short/repetitive sentences
    - 🚫 No artificial pause markers (model creates natural pauses)
    - ✅ Works exactly like original code but smarter
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter your text here...

The system will automatically:
- Merge sentences < 3 words with next sentence
- Merge repetitive text like "Há há há", "A a a"
- Keep everything else exactly as you typed

Example:
"Há há há!"
"A!"
"Cậu về rồi sao?"

→ Will become: "Há há há. A. Cậu về rồi sao?"
""", 
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
        use_smart_merge = gr.Checkbox(
            value=True,
            label="🧠 Enable Smart Merge",
            info="Merge short (<3 words) and repetitive sentences"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice", variant="primary", size="lg")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 📖 How It Works:
    
    | Step | Process |
    |------|---------|
    | 1️⃣ | **Smart Merge** (optional): Merges short/repetitive sentences |
    | 2️⃣ | **Post Processing**: Cleans up punctuation |
    | 3️⃣ | **TTSnorm**: Converts numbers/special chars to Vietnamese |
    | 4️⃣ | **Single-Pass TTS**: Reads entire text at once ✅ |
    
    ### 🎯 What Gets Merged:
    
    - ✅ Sentences with **< 3 words** (e.g., "A!", "Dạ?")
    - ✅ **Repetitive text** (e.g., "Há há há", "Hu hu hu")
    - ❌ Everything else stays **exactly as typed**
    
    **Merging uses only periods** - no extra words added!
    
    ### ⚙️ Technical Details:
    
    - Model: F5-TTS trained on ~1000 hours Vietnamese data
    - Processing: Single GPU pass (no chunking)
    - Pause control: Natural pauses at periods (model-based)
    - Max length: 1000 words per generation
    
    ### 📌 Why Single Pass?
    
    ✅ **More stable** - no audio concatenation issues  
    ✅ **Better flow** - natural prosody across entire text  
    ✅ **Fewer errors** - eliminates chunking problems  
    ✅ **Faster** - one model inference instead of many  
    
    **Note:** Pause duration is controlled by the model based on punctuation. 
    You cannot manually adjust silence length in single-pass mode.
    """)
    
    with gr.Accordion("❗ Limitations & Tips", open=False):
        gr.Markdown("""
        **Limitations:**
        1. Numbers/dates may not pronounce correctly (needs better normalization)
        2. Reference audio quality affects output quality
        3. Very long texts (>1000 words) may fail or produce poor quality
        4. Foreign words pronounced phonetically in Vietnamese
        
        **Tips for Best Results:**
        - Use clear reference audio (15-30s, no background noise)
        - Keep text under 1000 words
        - Use proper punctuation for natural pauses
        - Enable smart merge for dialogue-heavy text
        - Disable smart merge for poetry or carefully crafted text
        """)

    # Connect button
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, use_smart_merge], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
