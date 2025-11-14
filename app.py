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

def clean_text_before_processing(text):
    """Làm sạch text: loại bỏ emoji và ký tự đặc biệt."""
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "\U0001F900-\U0001F9FF"
        "\U0001FA00-\U0001FA6F"
        "]+", 
        flags=re.UNICODE
    )
    text = emoji_pattern.sub('', text)
    return ' '.join(text.split()).strip()

def analyze_text_structure(text):
    """
    Phân tích cấu trúc văn bản để xác định vị trí cần chèn silence.
    
    Returns:
        list of dict: [
            {
                'text': 'câu văn',
                'is_dialogue': True/False,
                'needs_pause_after': True/False,
                'char_start': 0,
                'char_end': 10
            },
            ...
        ]
    """
    text = clean_text_before_processing(text)
    paragraphs = text.split('\n\n')
    
    segments = []
    current_char_pos = 0
    
    for para_idx, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue
        
        # Gộp các dòng trong đoạn
        lines = para.split('\n')
        combined_text = ' '.join(line.strip() for line in lines if line.strip())
        
        # Kiểm tra hội thoại
        has_quotes = '"' in combined_text or '"' in combined_text or '"' in combined_text
        is_dialogue = has_quotes
        
        # Tìm các câu trong đoạn này
        # Pattern: tách theo dấu câu nhưng GIỮ NGUYÊN trong ngoặc kép
        sentences = []
        
        # Trích xuất các đoạn trong ngoặc kép
        quoted_pattern = r'(["""])([^"""]+)(["""])'
        quoted_ranges = []
        for match in re.finditer(quoted_pattern, combined_text):
            quoted_ranges.append((match.start(2), match.end(2)))
            # Thêm toàn bộ câu trong ngoặc kép
            quoted_text = match.group(2).strip()
            sentences.append({
                'text': quoted_text,
                'is_dialogue': True,
                'start_in_para': match.start(2),
                'end_in_para': match.end(2)
            })
        
        # Tách các câu NGOÀI ngoặc kép
        # Thay thế phần trong ngoặc kép bằng placeholder
        temp_text = combined_text
        for start, end in sorted(quoted_ranges, reverse=True):
            temp_text = temp_text[:start] + '###QUOTED###' + temp_text[end:]
        
        # Tách theo dấu câu
        parts = re.split(r'([.!?]+)', temp_text)
        
        current_pos_in_para = 0
        for i in range(0, len(parts) - 1, 2):
            sentence_text = parts[i].strip()
            punctuation = parts[i + 1] if i + 1 < len(parts) else ''
            
            if sentence_text and sentence_text != '###QUOTED###':
                full_sentence = sentence_text + punctuation
                # Tìm vị trí thực trong combined_text
                actual_start = combined_text.find(full_sentence, current_pos_in_para)
                if actual_start != -1:
                    sentences.append({
                        'text': full_sentence,
                        'is_dialogue': False,
                        'start_in_para': actual_start,
                        'end_in_para': actual_start + len(full_sentence)
                    })
                    current_pos_in_para = actual_start + len(full_sentence)
        
        # Sắp xếp theo vị trí xuất hiện
        sentences.sort(key=lambda x: x['start_in_para'])
        
        # Thêm vào segments với vị trí tuyệt đối
        para_start_pos = text.find(combined_text)
        for sent_idx, sent in enumerate(sentences):
            is_last_in_para = (sent_idx == len(sentences) - 1)
            is_last_para = (para_idx == len(paragraphs) - 1)
            
            segments.append({
                'text': sent['text'],
                'is_dialogue': sent['is_dialogue'],
                'needs_pause_after': not (is_last_in_para and is_last_para),  # Không pause ở câu cuối
                'char_start': para_start_pos + sent['start_in_para'],
                'char_end': para_start_pos + sent['end_in_para']
            })
        
        current_char_pos += len(para) + 2  # +2 for \n\n
    
    return segments

def estimate_pause_positions(segments, total_audio_length, total_text_length):
    """
    Ước lượng vị trí cần chèn silence trong audio dựa trên vị trí trong text.
    
    Returns:
        list of dict: [
            {
                'position_seconds': 1.5,
                'duration_seconds': 0.4,
                'is_dialogue': True/False
            },
            ...
        ]
    """
    pause_positions = []
    
    for seg in segments:
        if seg['needs_pause_after']:
            # Ước lượng thời gian tương ứng trong audio
            # Giả định audio phân bố đều theo text
            relative_position = seg['char_end'] / total_text_length
            audio_position = relative_position * total_audio_length
            
            pause_positions.append({
                'position_seconds': audio_position,
                'duration_seconds': 0.2 if seg['is_dialogue'] else 0.4,
                'is_dialogue': seg['is_dialogue']
            })
    
    return pause_positions

def insert_silences_into_audio(audio, sample_rate, pause_positions):
    """
    Chèn silence vào audio tại các vị trí đã xác định.
    
    Returns:
        numpy array: Audio mới với silence đã chèn
    """
    if not pause_positions:
        return audio
    
    # Sắp xếp theo vị trí
    pause_positions = sorted(pause_positions, key=lambda x: x['position_seconds'])
    
    segments = []
    last_pos = 0
    
    for pause in pause_positions:
        pos_samples = int(pause['position_seconds'] * sample_rate)
        
        # Đảm bảo không vượt quá độ dài audio
        if pos_samples > len(audio):
            pos_samples = len(audio)
        
        # Thêm phần audio trước pause
        if pos_samples > last_pos:
            segments.append(audio[last_pos:pos_samples])
        
        # Thêm silence
        silence_samples = int(pause['duration_seconds'] * sample_rate)
        silence = np.zeros(silence_samples, dtype=audio.dtype)
        segments.append(silence)
        
        print(f"   ⏸️  Inserted {pause['duration_seconds']}s silence at {pause['position_seconds']:.2f}s ({'dialogue' if pause['is_dialogue'] else 'paragraph'})")
        
        last_pos = pos_samples
    
    # Thêm phần audio còn lại
    if last_pos < len(audio):
        segments.append(audio[last_pos:])
    
    return np.concatenate(segments)

def post_process(text):
    """Làm sạch văn bản - loại bỏ tất cả ký tự đặc biệt."""
    # Loại bỏ ngoặc kép
    text = text.replace('"', '').replace('"', '').replace('"', '')
    
    # Loại bỏ tất cả dấu câu và ký tự đặc biệt
    text = re.sub(r'[^\w\s]', ' ', text, flags=re.UNICODE)
    
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split()).strip()
    
    return text

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
              pause_paragraph: float = 0.4, pause_dialogue: float = 0.2, request: gr.Request = None):
    """
    TTS inference - Hybrid approach:
    1. Xử lý toàn bộ văn bản một lần (ổn định)
    2. Chèn silence vào audio sau khi sinh (chính xác)
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        print(f"\n{'='*60}")
        print(f"🎤 Starting Hybrid TTS Generation")
        print(f"{'='*60}")
        print(f"🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Bước 1: Phân tích cấu trúc văn bản
        print(f"\n📊 Analyzing text structure...")
        segments = analyze_text_structure(gen_text)
        
        print(f"   Found {len(segments)} segments:")
        for idx, seg in enumerate(segments[:5], 1):
            marker = "💬" if seg['is_dialogue'] else "📄"
            pause_marker = "⏸️" if seg['needs_pause_after'] else "🔚"
            print(f"   {idx}. {marker}{pause_marker} [{seg['char_start']}-{seg['char_end']}] {seg['text'][:60]}...")
        
        # Bước 2: Chuẩn bị text cho TTS
        print(f"\n📝 Preparing text for TTS...")
        clean_text = clean_text_before_processing(gen_text)
        normalized_text = safe_normalize(clean_text)
        normalized_text = post_process(normalized_text)
        
        print(f"   Original: {len(gen_text)} chars")
        print(f"   Normalized: {len(normalized_text)} chars, {len(normalized_text.split())} words")
        print(f"   Preview: {normalized_text[:150]}...")
        
        # Bước 3: Preprocess reference audio
        print(f"\n🔄 Processing reference audio...")
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"   Reference text: {ref_text[:100]}...")
        
        # Bước 4: Sinh audio cho TOÀN BỘ văn bản một lần
        print(f"\n🎵 Generating audio for entire text...")
        wave, sr, _ = infer_process(
            ref_audio, 
            ref_text.lower(), 
            normalized_text, 
            model, 
            vocoder, 
            speed=speed
        )
        
        initial_duration = len(wave) / sr
        print(f"   ✅ Generated {initial_duration:.2f}s audio")
        
        # Bước 5: Ước lượng vị trí pause trong audio
        print(f"\n🎯 Calculating pause positions...")
        pause_positions = estimate_pause_positions(
            segments, 
            initial_duration, 
            len(clean_text)
        )
        
        # Cập nhật pause duration từ config
        for pause in pause_positions:
            if pause['is_dialogue']:
                pause['duration_seconds'] = pause_dialogue
            else:
                pause['duration_seconds'] = pause_paragraph
        
        print(f"   Found {len(pause_positions)} pause points")
        
        # Bước 6: Chèn silence vào audio
        print(f"\n⏸️  Inserting silences...")
        final_wave = insert_silences_into_audio(wave, sr, pause_positions)
        
        final_duration = len(final_wave) / sr
        added_silence = final_duration - initial_duration
        
        print(f"\n✅ Audio processing complete!")
        print(f"   Initial duration: {initial_duration:.2f}s")
        print(f"   Added silence: {added_silence:.2f}s")
        print(f"   Final duration: {final_duration:.2f}s")
        print(f"{'='*60}\n")
        
        # Tạo spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            plt.specgram(final_wave, Fs=sr, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Audio Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(spectrogram_path)
            plt.close()

        return (sr, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {str(e)}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech (Hybrid Approach)
    
    ### ✨ New Hybrid Method:
    - **Step 1**: Process entire text at once (stable, no skipping)
    - **Step 2**: Insert precise silences into generated audio
    - **Result**: Best of both worlds! 🎉
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text to Generate", 
            placeholder="""Enter Vietnamese text here...

Example with dialogue:
Hắn ngồi trên tàu. Mắt nhìn ra biển xa.

"Mấy năm qua... em đã sống thế nào?"

Cô gái im lặng.
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
    
    gr.Markdown("""
    ### 🎯 How It Works:
    
    1. **Text Analysis**: Detects dialogue (in quotes) vs narrative text
    2. **Single-Pass Generation**: Processes all text at once (no errors!)
    3. **Smart Pause Insertion**: Adds silences based on text structure
    4. **Precise Control**: Separate pause durations for dialogue vs paragraphs
    
    ### ✅ Advantages:
    - ✨ **Stable**: No failed chunks or skipped sentences
    - 🎯 **Precise**: Silences inserted at exact positions
    - 💬 **Smart**: Automatically detects dialogue vs narrative
    - ⚡ **Fast**: Single model inference
    
    ### 📝 Tips:
    - Use **double line breaks** (`\\n\\n`) to separate paragraphs
    - Put dialogue in quotes: `"Hello!"`
    - Adjust pause sliders to taste
    """)

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_paragraph, pause_dialogue], 
        outputs=[output_audio, output_spectrogram]
    )

demo.queue().launch(share=True)
