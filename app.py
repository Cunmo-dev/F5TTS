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
    Tách văn bản thành các câu với thông tin pause.
    
    Returns:
        list of tuples: [(sentence, pause_after_in_seconds, sentence_group_id), ...]
        - Các câu cùng group sẽ được merge audio với silence giữa chúng
    """
    chunks = []
    
    # Tách theo dòng trống
    paragraphs = text.split('\n\n')
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Xác định loại nội dung
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        open_quotes = combined_text.count('"') + combined_text.count('"')
        close_quotes = combined_text.count('"') + combined_text.count('"')
        is_dialogue = (open_quotes > 0 and open_quotes == close_quotes)
        pause_duration = pause_dialogue_duration if is_dialogue else pause_paragraph_duration
        
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
                    chunks.append((sentence_text, pause_duration, None))
                    current_sentence = ""
        
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration, None))
    
    # Gom nhóm các câu ngắn - chúng sẽ được generate riêng nhưng ghép với silence
    processed_chunks = []
    temp_group = []
    group_id = 0
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, _) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        
        if word_count >= 5:
            # Câu dài: xuất group trước (nếu có), rồi thêm câu này
            if temp_group:
                for s, p in temp_group:
                    processed_chunks.append((s, p, group_id))
                group_id += 1
                temp_group = []
            
            # Thêm câu dài (độc lập)
            processed_chunks.append((sentence, pause, group_id))
            group_id += 1
        else:
            # Câu ngắn: thêm vào group
            temp_group.append((sentence, pause))
            temp_pause = pause
            
            # Kiểm tra xem có nên xuất group không
            total_words = sum(len(s.split()) for s in [s for s, _ in temp_group])
            should_output = total_words >= 5
            
            if should_output or is_last:
                for s, p in temp_group:
                    processed_chunks.append((s, p, group_id))
                group_id += 1
                temp_group = []
    
    # Xử lý group còn sót
    if temp_group:
        for s, p in temp_group:
            processed_chunks.append((s, p, group_id))
    
    # Log thông tin
    print(f"\n📦 Grouped {len(processed_chunks)} sentences into {len(set(gid for _, _, gid in processed_chunks))} groups")
    current_group = None
    for sentence, pause, gid in processed_chunks[:10]:
        if gid != current_group:
            print(f"\n   Group {gid}:")
            current_group = gid
        print(f"      - [{len(sentence.split())}w, {pause}s] {sentence[:60]}...")
    
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
    TTS với pause thực sự - Generate riêng từng câu, ghép audio với silence.
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cấu hình pause
        pause_configs = {
            "Short": {
                "paragraph": 0.3,
                "dialogue": 0.15,
                "within_group": 0.15  # Pause giữa các câu trong cùng group
            },
            "Medium": {
                "paragraph": 0.5,
                "dialogue": 0.25,
                "within_group": 0.3
            },
            "Long": {
                "paragraph": 0.8,
                "dialogue": 0.4,
                "within_group": 0.5
            }
        }
        
        config = pause_configs.get(pause_level, pause_configs["Medium"])
        
        print(f"\n🎛️ Pause config: {config}")
        
        # Tách và gom nhóm câu
        chunks = split_text_into_sentences(
            gen_text, 
            config["paragraph"], 
            config["dialogue"]
        )
        
        if not chunks:
            raise gr.Error("No valid sentences found in text. Please check your input.")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu với pause giữa các câu
        audio_segments = []
        sample_rate = 24000
        
        current_group = None
        for i, (sentence, pause_duration, group_id) in enumerate(chunks):
            is_new_group = (group_id != current_group)
            is_last_in_group = (i == len(chunks) - 1 or chunks[i+1][2] != group_id)
            
            print(f"\n🔄 [{i+1}/{len(chunks)}] Group {group_id}: {sentence[:60]}...")
            
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
                    
                    # Thêm silence dựa trên vị trí
                    if not is_last_in_group:
                        # Giữa các câu trong cùng group: pause ngắn
                        silence = create_silence(config["within_group"], sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️  Within-group pause: {config['within_group']}s")
                    elif i < len(chunks) - 1:
                        # Giữa các group: pause dài
                        silence = create_silence(pause_duration, sample_rate)
                        audio_segments.append(silence)
                        print(f"   ⏸️⏸️  Between-group pause: {pause_duration}s")
                        
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
            
            current_group = group_id
        
        # Ghép tất cả audio lại
        if not audio_segments:
            raise gr.Error("No valid audio segments generated. Please check your text or try simpler sentences.")
            
        final_wave = np.concatenate(audio_segments)
        
        # Tính số group
        num_groups = len(set(gid for _, _, gid in chunks))
        print(f"\n✅ Final audio: {len(final_wave)/sample_rate:.2f}s from {len(chunks)} sentences in {num_groups} groups")
        
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
    ### 💡 How Smart Pause Works (NEW IMPROVED VERSION):
    
    | Feature | Description |
    |---------|-------------|
    | **Individual Generation** | Each sentence generated separately for maximum control |
    | **Smart Grouping** | Short sentences grouped together with mini-pauses |
    | **Within-Group Pauses** | 0.15-0.5s between sentences in same group |
    | **Between-Group Pauses** | 0.3-0.8s between different groups |
    | **Three Levels** | Short/Medium/Long affect both within and between pauses |
    
    ### 📖 Usage Tips:
    - **Separate paragraphs** with double line breaks (`\n\n`)
    - **Short sentences** (< 5 words) are grouped together
    - **Within-group**: Smooth flow between related short sentences
    - **Between-group**: Clear separation between different thoughts
    - **Pause Levels**:
      - **Short**: Fast reading (0.15s/0.3s) - news, announcements
      - **Medium**: Natural storytelling (0.3s/0.5s) - recommended
      - **Long**: Dramatic reading (0.5s/0.8s) - audiobooks, poetry
    
    ### 🎯 Example Processing:
    ```
    Input:
    "À!"
    "Còn quýt?"
    "Nhà chồng em!"
    
    Hắn ngồi im. Mắt nhìn xa.
    
    → Processing:
    Group 0 (short sentences):
      - Generate "À!" → +0.3s silence
      - Generate "Còn quýt?" → +0.3s silence
      - Generate "Nhà chồng em!" → +0.5s silence (end of group)
    
    Group 1:
      - Generate "Hắn ngồi im" → +0.3s silence
      - Generate "Mắt nhìn xa" → +0.5s silence (end of group)
    
    Result: Perfect pause control!
    ```
    
    ### ⚠️ Note:
    - Each sentence generated independently then combined with precise silence
    - Longer texts take more time but give perfect pause quality
    - No dependency on model's internal pause behavior
    - Full control over every pause duration
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not pronounce dates/phone numbers correctly
        2. **Audio Quality**: Use clear reference audio without background noise
        3. **Reference Text**: Auto-transcribed with Whisper (may have errors)
        4. **Processing Time**: Increases with text length (sentence-by-sentence processing)
        5. **Foreign Words**: Pronounced phonetically in Vietnamese (e.g., "Merci" → "Mét-xi")
        6. **Very Short Sentences**: Automatically grouped with nearby sentences
        7. **Error Recovery**: If one sentence fails, processing continues with remaining text
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
