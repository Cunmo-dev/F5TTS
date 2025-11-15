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

def post_process(text, silence_duration=0.3):
    """
    Xử lý văn bản với các quy tắc thông minh:
    1. Phát hiện các câu hội thoại/câu độc lập bị ghép nhầm bằng dấu phẩy
    2. Giữ dấu phẩy cho các câu có ngữ nghĩa liên kết
    3. Loại bỏ ký tự đặc biệt
    4. Thêm khoảng lặng giữa các câu
    """
    
    # Danh sách các từ/cụm từ thường là câu độc lập (hội thoại, cảm thán)
    independent_phrases = [
        # 1. Tiếng cảm thán / Kêu gọi
        r'\bà\b', r'\bờ\b', r'\bư\b', r'\bừ\b', r'\ba\b',
        r'\bai\b', r'\bơi\b', r'\bụi\b', r'\bê\b',
        
        # 2. Đáp lời lịch sự
        r'\bdạ\b', r'\bvâng\b', r'\bạ\b', r'\bơ\b',
        r'\bdạ\s+(cậu|anh|chị|má|ba|ông|bà)',
        
        # 3. Câu trả lời ngắn đơn độc
        r'^\s*(không|có|rồi|chưa|được|ừ|ờ)\s*$',
        r'^\s*(đúng|sai|phải|nào|thôi|đi)\s*$',
        
        # 4. Ngăn cản / Yêu cầu dừng lại
        r'\bkhoan\s+(đã|lại|nào)\b', r'\bđợi\s+(đã|chút|tí|tý)\b',
        r'\bđừng\b', r'\bchậm\s+lại\b', r'\bdừng\s+lại\b',
        
        # 5. Thúc giục / Ra lệnh ngắn
        r'\bmau\b', r'\bnhanh\b', r'\bchạy\b', r'\bđi\s+nhanh\b',
        r'\bmau\s+lên\b', r'\bnhanh\s+lên\b',
        
        # 6. Hỏi ngắn (câu hỏi tag)
        r'\bsao\b', r'\bvậy\s+sao\b', r'\bthế\s+nào\b',
        r'\bhay\s+sao\b', r'\bphải\s+(không|chăng)\b',
        r'\bchứ\b', r'\bnhỉ\b', r'\bnhé\b',
        
        # 7. Xác nhận / Đồng ý
        r'^\s*được\s+(rồi|lắm|thôi|đó)\s*$',
        r'^\s*tốt\s+(rồi|lắm|thôi)\s*$',
        r'^\s*(ừ|ờ|uhm|uh)\s+(nhỉ|nhé|à)?\s*$',
        
        # 8. Lời chào / Từ biệt
        r'\bxin\s+chào\b', r'\btạm\s+biệt\b', r'\bchào\b',
        r'\bhẹn\s+gặp\s+lại\b', r'\bchúc\s+ngủ\s+ngon\b',
        
        # 9. Lời cảm ơn / Xin lỗi
        r'\bcảm\s+ơn\b', r'\bcám\s+ơn\b', r'\bxin\s+lỗi\b',
        r'\blàm\s+ơn\b', r'\bxin\s+cậu\b',
        
        # 10. Câu hỏi WH- ngắn
        r'^\s*(ai|gì|đâu|nào|sao|thế\s+nào|chi)\s*$',
        r'^\s*(cái\s+gì|làm\s+sao|thế\s+nào)\s*$',
        
        # 11. Tiếng kêu hét / Sợ hãi
        r'^\s*(á|a|ơi|úi|trời|chết|mẹ\s+ơi)\s*$',
        
        # 12. Câu ngắn với từ gọi (vocative)
        r'\b(cậu|anh|chị|em|má|ba|ông|bà)\s+(ơi|à|ạ|nhé)\s*$',
        
        # 13. Hỏi ngắn với động từ
        r'^\s*(có\s+phải|có\s+được|có\s+thể)\s+.{0,15}\s+(không|chăng|hay\s+sao)\s*$',
    ]
    
    # Đánh dấu các đoạn text trong dấu ngoặc kép
    quoted_sections = []
    placeholder_pattern = "<<<QUOTED_{}>>>"
    
    def save_quoted_text(match):
        index = len(quoted_sections)
        quoted_sections.append(match.group(0))
        return placeholder_pattern.format(index)
    
    text = re.sub(r'"[^"]*"', save_quoted_text, text)
    
    # ===== XỬ LÝ THÔNG MINH DẤU PHẨY =====
    def smart_comma_split(sentence):
        """
        Tách dấu phẩy thành dấu chấm nếu:
        - Sau dấu phẩy là câu độc lập (hội thoại, cảm thán)
        - Trước hoặc sau dấu phẩy có placeholder (ngoặc kép)
        """
        parts = sentence.split(',')
        
        if len(parts) <= 1:
            return [sentence]
        
        result = []
        current = parts[0].strip()
        
        for i in range(1, len(parts)):
            next_part = parts[i].strip()
            is_independent = False
            
            # 1. Kiểm tra các từ/cụm từ độc lập
            for pattern in independent_phrases:
                if re.search(pattern, next_part, re.IGNORECASE):
                    is_independent = True
                    break
            
            # 2. Kiểm tra nếu có placeholder (ngoặc kép)
            if '<<<QUOTED_' in current or '<<<QUOTED_' in next_part:
                is_independent = True
            
            # 3. Kiểm tra nếu phần tiếp theo quá ngắn và không có từ liên kết
            linking_words = ['của', 'và', 'với', 'cho', 'bởi', 'là', 'ở', 'tại', 'trong', 
                           'ngoài', 'trên', 'dưới', 'theo', 'nhưng', 'mà', 'thì', 'nên']
            
            if len(next_part) < 20 and not any(word in next_part.lower() for word in linking_words):
                transition_verbs = ['nói', 'hỏi', 'kêu', 'gọi', 'la', 'thét', 'rằng', 'là']
                if not any(current.strip().endswith(v) for v in transition_verbs):
                    is_independent = True
            
            # 4. Kiểm tra pattern đặc biệt
            if current.strip().endswith(':') or current.strip().endswith('"'):
                is_independent = True
            
            if is_independent:
                if current:
                    result.append(current)
                current = next_part
            else:
                current += ', ' + next_part
        
        if current:
            result.append(current)
        
        return result
    
    # Tách câu theo dấu chấm
    sentences_by_period = text.split('.')
    
    all_sentences = []
    for sent in sentences_by_period:
        sent = sent.strip()
        if not sent:
            continue
        
        sub_sentences = smart_comma_split(sent)
        all_sentences.extend(sub_sentences)
    
    # Loại bỏ ký tự đặc biệt
    special_chars_pattern = r'[!@#$%^&*()_+=\[\]{};:\\|<>/?~`"\']'
    
    processed_sentences = []
    for sentence in all_sentences:
        if not sentence:
            continue
        
        if '<<<QUOTED_' not in sentence:
            sentence = re.sub(special_chars_pattern, '', sentence)
        
        sentence = sentence.strip()
        if sentence:
            processed_sentences.append(sentence)
    
    # Xử lý các đoạn text trong dấu ngoặc kép
    def process_quoted_text(quoted_with_marks):
        quoted = quoted_with_marks.strip('"')
        special_chars_pattern_quote = r'[!@#$%^&*()_+=\[\]{};:\\|<>/?~`]'
        quoted = re.sub(special_chars_pattern_quote, '', quoted)
        quoted = " ".join(quoted.split())
        return '"' + quoted + '"'
    
    # Khôi phục và xử lý các đoạn text trong ngoặc kép
    final_sentences = []
    for sentence in processed_sentences:
        for i, quoted_section in enumerate(quoted_sections):
            placeholder = placeholder_pattern.format(i)
            if placeholder in sentence:
                processed_quoted = process_quoted_text(quoted_section)
                sentence = sentence.replace(placeholder, processed_quoted)
        
        final_sentences.append(sentence)
    
    # Nối các câu lại với silence marker
    if silence_duration > 0:
        num_dots = int(silence_duration * 10)
        silence_marker = "." * num_dots
        text = silence_marker.join(final_sentences) + "."
    else:
        text = ". ".join(final_sentences) + "."
    
    # Loại bỏ khoảng trắng thừa
    text = " ".join(text.split())
    
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
def infer_tts(ref_audio_orig: str, gen_text: str, speed: float = 1.0, silence_duration: float = 0.3, request: gr.Request = None):
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Xử lý text với silence duration
        processed_text = post_process(TTSnorm(gen_text), silence_duration).lower()
        
        # Chạy inference
        final_wave, final_sample_rate, spectrogram = infer_process(
            ref_audio, ref_text.lower(), processed_text, model, vocoder, speed=speed
        )
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (final_sample_rate, final_wave), spectrogram_path
    except Exception as e:
        raise gr.Error(f"Error generating voice: {e}")

# Gradio UI
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🎤 F5-TTS: Vietnamese Text-to-Speech Synthesis
    # The model was trained with approximately 1000 hours of data on a RTX 3090 GPU
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
5. Smart comma handling: splits independent phrases (interjections, short responses) while preserving commas in contextual sentences.""", 
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
