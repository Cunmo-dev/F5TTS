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
    Tách văn bản thành các câu, KHÔNG BỎ QUA bất kỳ câu nào (kể cả câu ngắn).
    
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
                
                # THAY ĐỔI: Chỉ bỏ qua câu hoàn toàn rỗng, KHÔNG bỏ qua câu ngắn
                if sentence_text:
                    chunks.append((sentence_text, pause_duration))
                    current_sentence = ""
        
        # Thêm phần còn lại nếu có (kể cả câu ngắn)
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    # THAY ĐỔI: Gộp thông minh hơn - CHỈ gộp các câu CỰC NGẮN (< 3 từ)
    # và KHÔNG gộp nếu câu có dấu chấm than hoặc chấm hỏi (thường là câu độc lập)
    merged_chunks = []
    temp_sentence = ""
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        has_strong_punct = sentence.rstrip().endswith(('!', '?'))
        
        # LOGIC MỚI: Chỉ gộp nếu:
        # 1. Câu CỰC NGẮN (< 3 từ)
        # 2. KHÔNG có dấu chấm than/hỏi (câu độc lập)
        # 3. KHÔNG phải câu cuối
        should_merge = (word_count < 3) and (not has_strong_punct) and (not is_last)
        
        if should_merge and temp_sentence:
            # Gộp với câu trước
            temp_sentence += " " + sentence
        elif should_merge:
            # Bắt đầu tích lũy
            temp_sentence = sentence
            temp_pause = pause
        else:
            # Câu đủ dài hoặc có dấu mạnh -> xuất luôn
            if temp_sentence:
                # Xuất câu tích lũy trước
                merged_chunks.append((temp_sentence + " " + sentence, pause))
                temp_sentence = ""
            else:
                # Xuất câu hiện tại
                merged_chunks.append((sentence, pause))
    
    # Xuất phần cuối nếu còn
    if temp_sentence:
        merged_chunks.append((temp_sentence, temp_pause))
    
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
    BẢN CẢI TIẾN: Đọc HẾT mọi câu, kể cả câu ngắn và ngoại ngữ.
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
        for idx, (sent, pause) in enumerate(chunks, 1):
            print(f"   {idx}. [{pause}s] {sent[:80]}...")
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        processed_count = 0
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 [{i+1}/{len(chunks)}] Processing: {sentence[:80]}...")
            
            # Chuẩn hóa văn bản - QUAN TRỌNG: Bọc try-except để xử lý ngoại ngữ
            try:
                normalized_text = post_process(TTSnorm(sentence)).lower()
            except Exception as norm_error:
                # Nếu TTSnorm fail (có thể do ngoại ngữ), dùng văn bản gốc
                print(f"   ⚠️  TTSnorm failed, using original text")
                normalized_text = post_process(sentence).lower()
            
            # THAY ĐỔI: Chấp nhận cả câu rất ngắn (>= 1 từ thay vì >= 5 từ)
            if len(normalized_text.strip().split()) < 1:
                print(f"   ⏭️ Skipped (empty): '{normalized_text}'")
                continue
            
            print(f"   📝 Normalized: {normalized_text[:80]}...")
            
            try:
                # Tạo audio cho câu này
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
                processed_count += 1
                print(f"   ✅ Generated {len(wave)/sr:.2f}s audio")
                
                # Thêm khoảng im lặng (trừ câu cuối)
                if i < len(chunks) - 1:
                    silence = create_silence(pause_duration, sample_rate)
                    audio_segments.append(silence)
                    print(f"   ⏸️  Added {pause_duration}s silence")
                    
            except Exception as e:
                print(f"   ❌ Error processing chunk: {e}")
                # THAY ĐỔI: Không bỏ qua hoàn toàn, thử xử lý đơn giản hơn
                try:
                    # Thử lần 2 với văn bản gốc không chuẩn hóa
                    simple_text = sentence.lower().strip()
                    wave, sr, _ = infer_process(
                        ref_audio, 
                        ref_text.lower(), 
                        simple_text, 
                        model, 
                        vocoder, 
                        speed=speed
                    )
                    sample_rate = sr
                    audio_segments.append(wave)
                    processed_count += 1
                    print(f"   ✅ Retry successful with simple text")
                    
                    if i < len(chunks) - 1:
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                except:
                    print(f"   ❌ Retry also failed, skipping...")
                    continue
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text.")
            
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s ({processed_count}/{len(chunks)} segments)")
        
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
            plt.title(f'Audio Spectrogram ({processed_count}/{len(chunks)} segments)')
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
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis (Enhanced)
    ### Model trained with ~1000 hours of data on RTX 3090 GPU
    
    ✨ **NEW**: Now reads ALL sentences including short phrases and foreign words!
    
    🌍 **Multilingual Support**: Handles Vietnamese mixed with English, French, etc.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter text with paragraphs and dialogue...

Example with mixed languages:
Hắn lúc này đang ngồi trên boong tàu. Mắt nhìn ra biển xa.

"Toa lần này trở về nhà chơi được bao lâu?"

Người hỏi là một người bạn tình cờ gặp.

"Meci beaucoup!"

"Thank you very much!"
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
    ### 🆕 What's New:
    
    ✅ **All Sentences Read**: No more skipped short sentences!  
    ✅ **Foreign Words**: Handles "Merci beaucoup", "Thank you", etc.  
    ✅ **Smart Merging**: Only merges VERY short phrases (< 3 words)  
    ✅ **Punctuation Respect**: Never merges sentences with `!` or `?`  
    ✅ **Fallback System**: Double-check if normalization fails  
    
    ### 💡 How Smart Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Paragraph Detection** | Separates narrative text by double line breaks |
    | **Dialogue Detection** | Identifies quoted speech (even multi-line) |
    | **Real Silence** | Actual silent gaps (no fake sounds!) |
    | **Intelligent Merging** | Only combines extremely short fragments |
    | **Three Levels** | Short (0.2s/0.1s), Medium (0.4s/0.2s), Long (0.6s/0.3s) |
    
    ### 📖 Usage Tips:
    - **Separate paragraphs** with double line breaks (`\\n\\n`)
    - **Dialogue** can span multiple lines - just use quotes `"..."`
    - **Foreign words** are now supported (but pronunciation may vary)
    - **Short exclamations** like "Wow!" or "Oui!" will be read separately
    - **Short**: Fast-paced reading (news, announcements)
    - **Medium**: Natural storytelling (recommended)
    - **Long**: Dramatic audiobooks, poetry
    
    ### 🎯 Example Input:
    ```
    Hắn ngồi trên boong tàu. Mắt nhìn ra biển.
    
    "Toa lần này trở về nhà chơi được bao lâu?"
    
    Người hỏi là bạn từ Sài Gòn. 
    
    "Merci beaucoup!"
    
    Họ gặp nhau trên đất Pháp.
    ```
    
    ✨ **ALL** of these sentences will be read, including "Merci beaucoup!"
    
    ### ⚙️ How It Works:
    1. **Text Splitting**: Breaks text at `.`, `!`, `?` marks
    2. **Smart Filtering**: Only skips truly empty sentences
    3. **Minimal Merging**: Only combines fragments < 3 words WITHOUT `!` or `?`
    4. **Fallback Processing**: If TTSnorm fails, uses original text
    5. **Audio Assembly**: Concatenates all segments with real silence gaps
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
        5. **Foreign Words**: Will attempt to pronounce but may not sound native
        6. **Very Short Sentences**: Now processed but may sound unnatural if standalone
        
        ### 🔧 Technical Changes:
        - Removed strict word-count filters (was >= 5 words, now >= 1 word)
        - Added fallback for TTSnorm failures (helps with foreign text)
        - Improved merge logic to preserve independent exclamations
        - Added retry mechanism for failed segments
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
