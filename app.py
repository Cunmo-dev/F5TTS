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

def post_process(text):
    """
    Xử lý văn bản với các quy tắc:
    1. Thay thế tất cả dấu phẩy bằng dấu chấm để mỗi câu ngắn đều độc lập
    2. Câu trong dấu ngoặc kép "" được coi là câu riêng biệt
    3. Thêm dấu chấm trước dấu ngoặc kép mở
    4. Loại bỏ ký tự đặc biệt trong dấu ngoặc kép
    5. Nếu ký tự đặc biệt ở cuối câu trong ngoặc kép, thay bằng dấu chấm
    6. Loại bỏ dấu phẩy/chấm trùng lặp trong ngoặc kép
    7. Xử lý câu ngoài dấu ngoặc kép: loại bỏ ký tự đặc biệt cuối câu và thêm dấu chấm
    """
    
    # Đánh dấu các đoạn text trong dấu ngoặc kép để tránh xử lý nhầm
    quoted_sections = []
    placeholder_pattern = "<<<QUOTED_{}>>>"
    
    def save_quoted_text(match):
        index = len(quoted_sections)
        quoted_sections.append(match.group(0))
        return placeholder_pattern.format(index)
    
    # Tạm thời thay thế các đoạn text trong ngoặc kép bằng placeholder
    text = re.sub(r'"[^"]*"', save_quoted_text, text)
    
    # THAY TẤT CẢ DẤU PHẨY BẰNG DẤU CHẤM (ngoài dấu ngoặc kép)
    text = text.replace(',', '.')
    
    # Xử lý text ngoài dấu ngoặc kép
    # Tách thành các phần dựa trên placeholder
    parts = re.split(r'(<<<QUOTED_\d+>>>)', text)
    
    processed_parts = []
    special_chars_pattern = r'[!@#$%^&*()_+=\[\]{};:\\|<>/?~`"\']'
    
    for part in parts:
        if part.startswith('<<<QUOTED_') and part.endswith('>>>'):
            # Đây là placeholder, giữ nguyên
            processed_parts.append(part)
        else:
            # Xử lý text ngoài ngoặc kép
            if part.strip():
                # Loại bỏ ký tự đặc biệt ở cuối câu
                part = part.rstrip()
                if part:
                    # Loại bỏ tất cả ký tự đặc biệt ở cuối (không bao gồm dấu chấm)
                    while part and re.search(special_chars_pattern + r'$', part):
                        part = re.sub(special_chars_pattern + r'$', '', part).rstrip()
                    
                    # Thêm dấu chấm nếu chưa có
                    if part and not part.endswith('.'):
                        part += '.'
            
            processed_parts.append(part)
    
    text = ''.join(processed_parts)
    
    # Xử lý các đoạn text trong dấu ngoặc kép
    def process_quoted_text(quoted_with_marks):
        # Loại bỏ dấu ngoặc kép để xử lý nội dung
        quoted = quoted_with_marks.strip('"')
        
        # Loại bỏ các ký tự đặc biệt (giữ lại chữ cái, số, khoảng trắng, dấu phẩy và dấu chấm)
        special_chars_pattern = r'[!@#$%^&*()_+=\[\]{};:\\|<>/?~`]'
        quoted = re.sub(special_chars_pattern, '', quoted)
        
        # Loại bỏ dấu phẩy và dấu chấm trùng lặp (chỉ giữ 1 dấu)
        quoted = re.sub(r'\.{2,}', '.', quoted)  # Nhiều dấu chấm -> 1 dấu chấm
        quoted = re.sub(r',{2,}', ',', quoted)   # Nhiều dấu phẩy -> 1 dấu phẩy
        quoted = re.sub(r'[,\s]+\.', '.', quoted)  # Dấu phẩy + dấu chấm -> dấu chấm
        quoted = re.sub(r'\.[,\s]+', '. ', quoted)  # Dấu chấm + dấu phẩy -> dấu chấm
        
        # Xử lý ký tự đặc biệt ở cuối câu trong ngoặc kép
        quoted = quoted.strip()
        
        # Nếu câu không kết thúc bằng dấu chấm, thêm dấu chấm
        if quoted and not quoted.endswith('.'):
            # Loại bỏ dấu phẩy cuối cùng nếu có
            if quoted.endswith(','):
                quoted = quoted[:-1].strip()
            quoted += '.'
        
        # Trả về với dấu chấm trước ngoặc kép mở
        return '. "' + quoted + '"'
    
    # Khôi phục các đoạn text trong ngoặc kép và xử lý chúng
    for i, quoted_section in enumerate(quoted_sections):
        placeholder = placeholder_pattern.format(i)
        processed_quoted = process_quoted_text(quoted_section)
        text = text.replace(placeholder, processed_quoted)
    
    # Xử lý các dấu chấm trùng lặp
    text = " " + text + " "
    text = text.replace(" . . ", " . ")
    text = " " + text + " "
    text = text.replace(" .. ", " . ")
    text = " " + text + " "
    # Loại bỏ pattern ". ." nhiều lần
    while " . . " in text:
        text = text.replace(" . . ", " . ")
    
    # Loại bỏ dấu chấm thừa ở đầu câu (nếu có)
    text = re.sub(r'^\.\s+', '', text.strip())
    
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())
    
    return text

def split_sentences(text):
    """
    Chia văn bản thành các câu dựa trên dấu chấm
    """
    # Chia theo dấu chấm, loại bỏ câu rỗng
    sentences = [s.strip() for s in text.split('.') if s.strip()]
    return sentences

def add_silence(audio_array, sample_rate, silence_duration):
    """
    Thêm khoảng lặng vào cuối audio
    audio_array: numpy array của audio
    sample_rate: tần số lấy mẫu
    silence_duration: thời gian lặng (giây)
    """
    silence_samples = int(sample_rate * silence_duration)
    silence = np.zeros(silence_samples, dtype=audio_array.dtype)
    return np.concatenate([audio_array, silence])

# Load models
vocoder = load_vocoder()
model = load_model(
    DiT,
    dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
    ckpt_path=str(cached_path("hf://thanhcong190693/F5TTSVN/model_last.pt")),
    vocab_file=str(cached_path("hf://thanhcong190693/F5TTSVN/config.json")),
)

@spaces.GPU
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, silence_duration: float = 0.0, request: gr.Request = None):

    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Xử lý văn bản
        processed_text = post_process(TTSnorm(gen_text)).lower()
        
        # Chia thành các câu
        sentences = split_sentences(processed_text)
        
        if not sentences:
            raise gr.Error("No valid sentences found after text processing.")
        
        # Tiền xử lý audio tham chiếu
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tổng hợp từng câu và nối lại
        all_waves = []
        all_spectrograms = []
        
        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue
                
            # Sinh audio cho câu hiện tại
            wave, sample_rate, spectrogram = infer_process(
                ref_audio, 
                ref_text.lower(), 
                sentence, 
                model, 
                vocoder, 
                speed=speed
            )
            
            # Thêm khoảng lặng nếu không phải câu cuối
            if i < len(sentences) - 1 and silence_duration > 0:
                wave = add_silence(wave, sample_rate, silence_duration)
            
            all_waves.append(wave)
            all_spectrograms.append(spectrogram)
        
        # Nối tất cả các audio lại
        final_wave = np.concatenate(all_waves)
        
        # Lưu spectrogram của câu đầu tiên (hoặc có thể tổng hợp)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(all_spectrograms[0], spectrogram_path)

        return (sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis.
    # The model was trained with approximately 1000 hours of data on a RTX 3090 GPU. 
    Enter text and upload a sample voice to generate natural speech.
    """)
    
    with gr.Row():
        ref_audio = gr.Audio(label="🔊 Sample Voice", type="filepath")
        gen_text = gr.Textbox(label="📝 Text", placeholder="Enter the text to generate voice...", lines=3)
    
    with gr.Row():
        speed = gr.Slider(0.3, 2.0, value=1.0, step=0.1, label="⚡ Speed")
        silence_duration = gr.Slider(0.0, 2.0, value=0.3, step=0.1, label="🔇 Silence Between Sentences (seconds)")
    
    btn_synthesize = gr.Button("🔥 Generate Voice")
    
    with gr.Row():
        output_audio = gr.Audio(label="🎧 Generated Audio", type="numpy")
        output_spectrogram = gr.Image(label="📊 Spectrogram")
    
    model_limitations = gr.Textbox(
        value="""1. This model may not perform well with numerical characters, dates, special characters, etc. => A text normalization module is needed.
2. The rhythm of some generated audios may be inconsistent or choppy => It is recommended to select clearly pronounced sample audios with minimal pauses for better synthesis quality.
3. Default, reference audio text uses the pho-whisper-medium model, which may not always accurately recognize Vietnamese, resulting in poor voice synthesis quality.
4. Inference with overly long paragraphs may produce poor results.
5. The silence slider adds pauses between sentences split by periods (.) for better audio clarity.""", 
        label="❗ Model Limitations",
        lines=5,
        interactive=False
    )

    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, silence_duration], 
        outputs=[output_audio, output_spectrogram]
    )

# Run Gradio with share=True to get a gradio.live link
demo.queue().launch(share=True)
