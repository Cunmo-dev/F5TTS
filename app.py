
import spaces
import os
from huggingface_hub import login
import gradio as gr
from cached_path import cached_path
import tempfile
from vinorm import TTSnorm
import numpy as np
import re

from f5_tts.model import DiT
from f5_tts.infer.utils_infer import (
    preprocess_ref_audio_text,
    load_vocoder,
    load_model,
    infer_process,
    save_spectrogram,
)

# Set matplotlib backend
os.environ["MPLBACKEND"] = "Agg"

# Retrieve token from secrets
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Log in to Hugging Face
if hf_token:
    login(token=hf_token)

def post_process(text):
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    text = text.replace(" , , ", " , ")
    text = " " + text + " "
    text = text.replace(" ,, ", " , ")
    text = " " + text + " "
    text = text.replace('"', "")
    return " ".join(text.split())

def split_sentences(text, max_words=50):
    """
    Tách văn bản thành các câu với xử lý đặc biệt cho hội thoại.
    Giới hạn mỗi câu không quá max_words từ để tránh quá tải model.
    """
    sentences = []
    
    # Tách theo xuống dòng trước
    lines = text.split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Nếu là hội thoại (bắt đầu bằng dấu ngoặc kép)
        if line.startswith('"') or line.startswith('"') or line.startswith('"'):
            # Chia nhỏ hội thoại dài nếu cần
            if len(line.split()) > max_words:
                chunks = split_by_length(line, max_words)
                sentences.extend(chunks)
            else:
                sentences.append(line)
        else:
            # Tách theo dấu chấm, chấm than, chấm hỏi trước
            parts = re.split(r'(?<=[.!?])\s+', line)
            
            for part in parts:
                if not part.strip():
                    continue
                
                # Nếu câu quá dài, tách thêm theo dấu phẩy
                if len(part.split()) > max_words:
                    sub_parts = re.split(r'(?<=[,;])\s+', part)
                    for sub in sub_parts:
                        if sub.strip() and len(sub.strip()) > 5:
                            # Nếu vẫn còn dài, chia theo số từ
                            if len(sub.split()) > max_words:
                                chunks = split_by_length(sub, max_words)
                                sentences.extend(chunks)
                            else:
                                sentences.append(sub.strip())
                else:
                    sentences.append(part.strip())
    
    # Lọc bỏ các câu quá ngắn (< 5 ký tự)
    sentences = [s for s in sentences if len(s.strip()) > 5]
    
    return sentences

def split_by_length(text, max_words):
    """
    Chia văn bản thành các đoạn nhỏ theo số từ tối đa.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    
    for word in words:
        current_chunk.append(word)
        if len(current_chunk) >= max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def add_silence(audio_array, sample_rate, duration_ms=500):
    """
    Thêm khoảng lặng vào cuối audio array.
    """
    silence_samples = int(sample_rate * duration_ms / 1000)
    silence = np.zeros(silence_samples, dtype=audio_array.dtype)
    return np.concatenate([audio_array, silence])

def is_dialogue(text):
    """
    Kiểm tra xem câu có phải là hội thoại không.
    """
    text = text.strip()
    return text.startswith('"') or text.startswith('"') or text.startswith('"')

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
              pause_paragraph: int = 800, pause_dialogue: int = 400, 
              request: gr.Request = None):
    """
    Args:
        pause_paragraph: khoảng lặng sau đoạn văn tả (ms)
        pause_dialogue: khoảng lặng sau câu hội thoại (ms)
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tách văn bản thành các câu (giới hạn 50 từ/câu để tránh quá tải)
        sentences = split_sentences(gen_text, max_words=50)
        
        print(f"\n=== DETECTED {len(sentences)} SENTENCES ===")
        for i, sent in enumerate(sentences):
            print(f"Sentence {i+1}: {sent[:80]}...")
        print("=" * 50)
        
        if len(sentences) == 0:
            raise gr.Error("No valid sentences found in the text.")
        
        # Khởi tạo danh sách để lưu audio
        audio_segments = []
        sample_rate = None
        all_spectrograms = []
        
        # Xử lý từng câu
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
            
            try:
                # Chuẩn hóa và xử lý câu
                processed_sentence = post_process(TTSnorm(sentence)).lower()
                
                print(f"Processing {i+1}/{len(sentences)}: {processed_sentence[:60]}...")
                
                # Tạo audio cho câu
                wave, sr, spectrogram = infer_process(
                    ref_audio, 
                    ref_text.lower(), 
                    processed_sentence, 
                    model, 
                    vocoder, 
                    speed=speed
                )
                
                if sample_rate is None:
                    sample_rate = sr
                
                # Thêm khoảng lặng phù hợp
                if i < len(sentences) - 1:
                    # Hội thoại: pause ngắn hơn
                    # Đoạn văn tả: pause dài hơn
                    pause = pause_dialogue if is_dialogue(sentence) else pause_paragraph
                    wave = add_silence(wave, sr, pause)
                
                audio_segments.append(wave)
                all_spectrograms.append(spectrogram)
                
            except Exception as e:
                print(f"Warning: Failed to process sentence {i+1}: {e}")
                # Tiếp tục với câu tiếp theo thay vì dừng hẳn
                continue
        
        if len(audio_segments) == 0:
            raise gr.Error("Failed to generate any audio segments.")
        
        # Ghép tất cả audio lại
        final_wave = np.concatenate(audio_segments)
        
        # Tạo spectrogram tổng hợp
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(all_spectrograms[0], spectrogram_path)

        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    # The model was trained with approximately 1000 hours of data on a RTX 3090 GPU
    Enter text and upload a sample voice to generate natural speech with intelligent pausing.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="📝 Text", 
            placeholder="Enter the text to generate voice (supports paragraphs and dialogue)...", 
            lines=8
        )
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        pause_paragraph = gr.Slider(
            200, 2000, value=800, step=100, 
            label="⏸️ Pause After Paragraph (ms)"
        )
        pause_dialogue = gr.Slider(
            100, 1500, value=400, step=50, 
            label="💬 Pause After Dialogue (ms)"
        )
    
    btn_synthesize = gr.Button("🔥 Generate Voice")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    gr.Markdown("""
    ### 💡 Tips:
    - **Paragraph Pause**: Longer silence after descriptive text (default 800ms)
    - **Dialogue Pause**: Shorter silence between dialogue lines (default 400ms)
    - System automatically detects dialogue (text in quotes) vs narration
    - For natural conversation flow, use 300-500ms for dialogue
    - For dramatic reading, increase paragraph pause to 1000-1500ms
    - **NEW**: Now splits sentences by commas (,) and semicolons (;) in addition to periods (.)
    """)
    
    model_limitations = gr.Textbox(
        value="""1. This model may not perform well with numerical characters, dates, special characters, etc. => A text normalization module is needed.
2. The rhythm of some generated audios may be inconsistent or choppy => It is recommended to select clearly pronounced sample audios with minimal pauses for better synthesis quality.
3. Default, reference audio text uses the pho-whisper-medium model, which may not always accurately recognize Vietnamese, resulting in poor voice synthesis quality.
4. Inference with overly long paragraphs may produce poor results.""", 
        label="❗ Model Limitations",
        lines=4,
        interactive=False
    )

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_paragraph, pause_dialogue], 
        outputs=[output_audio, output_spectrogram]
    )

# Run Gradio with share=True
demo.queue().launch(share=True)
