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

# Login HF
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if hf_token:
    login(token=hf_token)

# ==================== PAUSE MARKER SYSTEM ====================
PAUSE_MARKER = "|||"  # Ký hiệu đặc biệt - model sẽ đọc nhẹ hoặc im

def add_pause_markers(text, pause_level="Medium"):
    """
    Thay thế dấu câu bằng marker + giữ nguyên văn bản.
    """
    pause_configs = {
        "Short":   ("|||", "||"),     # Paragraph: 0.3s, Dialogue: 0.15s
        "Medium":  ("|||||", "|||"),  # Paragraph: 0.5s, Dialogue: 0.25s
        "Long":    ("|||||||", "|||||")  # Paragraph: 0.8s, Dialogue: 0.4s
    }
    para_marker, dia_marker = pause_configs.get(pause_level, ("|||||", "|||"))

    paragraphs = text.split('\n\n')
    processed = []

    for para in paragraphs:
        para = para.strip()
        if not para: continue

        lines = para.split('\n')
        combined = ' '.join(line.strip() for line in lines if line.strip())
        has_quotes = '"' in combined or '“' in combined or '”' in combined
        marker = dia_marker if has_quotes else para_marker

        # Thay .!? bằng marker
        combined = re.sub(r'([.!?])\s*', rf'\1 {marker} ', combined)
        processed.append(combined)

    result = '\n\n'.join(processed)
    print(f"Pause markers added: {result[:200]}...")
    return result

def post_process_text(text):
    """Chỉ làm sạch, KHÔNG xóa marker"""
    text = " " + text + " "
    text = text.replace('"', '').replace('“', '').replace('”', '')
    text = re.sub(r',+', ',', text)
    return " ".join(text.split())

def split_audio_by_marker(wave, sample_rate, marker_duration_sec=0.08, energy_threshold=1e-5):
    """
    Tách audio tại các đoạn im lặng (do model phát marker |||)
    """
    chunk_size = int(marker_duration_sec * sample_rate)
    min_segment_length = int(0.2 * sample_rate)  # ít nhất 0.2s mỗi đoạn
    segments = []
    current = []
    last_non_silent = 0

    for i in range(0, len(wave), chunk_size // 2):
        start = i
        end = min(i + chunk_size, len(wave))
        chunk = wave[start:end]
        energy = np.mean(chunk ** 2)

        if energy < energy_threshold:
            # Im lặng → có thể là marker
            if len(current) > 0 and sum(len(c) for c in current) > min_segment_length:
                # Đã đủ dài → cắt đoạn
                segment = np.concatenate(current)
                if len(segment) > min_segment_length:
                    segments.append(segment)
                current = []
                last_non_silent = end
        else:
            # Có âm → thêm vào current
            if len(current) == 0 and i > last_non_silent:
                # Bắt đầu đoạn mới - giữ dạng numpy array
                current.append(wave[max(0, i - chunk_size):end])
            else:
                current.append(chunk)
            last_non_silent = end

    # Thêm đoạn cuối
    if len(current) > 0 and sum(len(c) for c in current) > min_segment_length // 2:
        segment = np.concatenate(current)
        if len(segment) > 0:
            segments.append(segment)

    # Nếu không tách được → trả về 1 đoạn duy nhất
    if len(segments) == 0 and len(wave) > 0:
        segments.append(wave)

    return segments

def insert_silence_between(segments, silence_durations):
    """Ghép các đoạn với silence thật"""
    if len(segments) == 0:
        return np.array([], dtype=np.float32)
    result = [segments[0]]
    for i in range(1, len(segments)):
        silence_dur = silence_durations[min(i-1, len(silence_durations)-1)]
        silence = np.zeros(int(silence_dur * 24000), dtype=np.float32)
        result.extend([silence, segments[i]])
    return np.concatenate(result)


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
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")

    try:
        print(f"\n{'='*60}")
        print(f"Starting Hybrid TTS (Whole-text + Real Silence)")
        print(f"{'='*60}")

        # 1. Thêm pause marker
        text_with_markers = add_pause_markers(gen_text, pause_level)

        # 2. Chuẩn hóa
        normalized_text = post_process_text(TTSnorm(text_with_markers)).lower()

        # 3. Preprocess ref
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"Ref text: {ref_text[:100]}...")

        # 4. Sinh audio TOÀN BỘ MỘT LẦN
        wave, sr, spectrogram = infer_process(
            ref_audio, ref_text.lower(), normalized_text, model, vocoder, speed=speed
        )

       # Thay thế đoạn từ dòng ~140 trở đi:
        # 5. Tách audio tại marker
        marker_segments = split_audio_by_marker(wave, sr, marker_duration_sec=0.08, energy_threshold=1e-6)

        print(f"Detected {len(marker_segments)} audio segments")

        # 6. Tính silence duration
        silence_map = {
            "Short":  [0.3, 0.15],
            "Medium": [0.5, 0.25],
            "Long":   [0.8, 0.40]
        }
        para_s, dia_s = silence_map.get(pause_level, [0.5, 0.25])

        # Đếm số câu trong văn bản gốc
        sentences = re.findall(r'[.!?]+', gen_text)
        dialogue_sentences = len(re.findall(r'"[^"]*[.!?]', gen_text))
        narrative_sentences = len(sentences) - dialogue_sentences

        silence_durations = [para_s] * narrative_sentences + [dia_s] * dialogue_sentences
        silence_durations = silence_durations[:len(marker_segments)-1]

        # Nếu không đủ pause → dùng mặc định
        if len(silence_durations) < len(marker_segments) - 1:
            silence_durations = [para_s] * (len(marker_segments) - 1)

        # 7. Ghép lại
        final_wave = insert_silence_between(marker_segments, silence_durations)

        duration = len(final_wave) / sr
        print(f"Final audio: {duration:.2f}s | Segments: {len(marker_segments)}")

        # 8. Lưu spectrogram
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            spectrogram_path = f.name
            save_spectrogram(spectrogram, spectrogram_path)

        return (sr, final_wave), spectrogram_path

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error: {e}")

# ==================== GRADIO UI ====================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # F5-TTS Hybrid: **Toàn văn bản + Real Silence**
    ### Xử lý **toàn bộ một lần** + **pause thật**, **không bỏ sót từ**
    """)

    with gr.Row():
        ref_audio = gr.Audio(label="Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="Text to Generate", lines=10,
            placeholder="Hắn ngồi trên tàu. Mắt nhìn xa.\n\n\"Toa về bao lâu?\"\n\n\"Meci beaucoup!\""
        )

    with gr.Row():
        speed = gr.Slider(0.3, 2.0, 1.0, step=0.1, label="Speed")
        pause_level = gr.Radio(["Short", "Medium", "Long"], "Medium", label="Pause Level")

    btn = gr.Button("Generate Voice", variant="primary")
    with gr.Row():
        out_audio = gr.Audio(label="Generated Audio", type="numpy")
        out_spec = gr.Image(label="Spectrogram")

    gr.Markdown("""
    ### Ưu điểm:
    - **Không bỏ sót từ** (toàn văn bản 1 lần)
    - **Pause thật, không âm lạ** (silence ghép sau)
    - **Hỗ trợ từ nước ngoài**: "Meci beaucoup!" → đọc đúng
    - **Tốc độ nhanh hơn Code 2**, chất lượng pause tốt hơn Code 1
    """)

    btn.click(infer_tts, [ref_audio, gen_text, speed, pause_level], [out_audio, out_spec])

demo.queue().launch(share=True)
