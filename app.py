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
PAUSE_MARKER = "|||"

def add_pause_markers(text, pause_level="Medium"):
    """
    Thay thế dấu câu bằng marker (NGẮN HƠN để không chậm)
    """
    pause_configs = {
        "Short":   ("|", ""),        # Paragraph: 0.15s, Dialogue: 0s
        "Medium":  ("||", "|"),      # Paragraph: 0.3s, Dialogue: 0.15s
        "Long":    ("|||", "||")     # Paragraph: 0.5s, Dialogue: 0.3s
    }
    para_marker, dia_marker = pause_configs.get(pause_level, ("||", "|"))

    paragraphs = text.split('\n\n')
    processed = []

    for para in paragraphs:
        para = para.strip()
        if not para: continue

        lines = para.split('\n')
        combined = ' '.join(line.strip() for line in lines if line.strip())
        has_quotes = '"' in combined or '"' in combined or '"' in combined
        marker = dia_marker if has_quotes else para_marker

        # Thay .!? bằng marker NGẮN
        combined = re.sub(r'([.!?])\s*', rf'\1 {marker} ', combined)
        processed.append(combined)

    result = '\n\n'.join(processed)
    print(f"Pause markers added: {result[:200]}...")
    return result

def post_process_text(text):
    """Chỉ làm sạch, KHÔNG xóa marker"""
    text = " " + text + " "
    text = text.replace('"', '').replace('"', '').replace('"', '')
    text = re.sub(r',+', ',', text)
    return " ".join(text.split())

def split_audio_by_marker(wave, sample_rate, marker_duration_sec=0.05, energy_threshold=1e-5):
    """
    Tách audio tại các đoạn im lặng (marker_duration_sec GIẢM xuống 0.05s)
    """
    chunk_size = int(marker_duration_sec * sample_rate)
    min_segment_length = int(0.15 * sample_rate)  # Giảm từ 0.2s → 0.15s
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
                segment = np.concatenate(current)
                if len(segment) > min_segment_length:
                    segments.append(segment)
                current = []
                last_non_silent = end
        else:
            # Có âm → thêm vào current
            if len(current) == 0 and i > last_non_silent:
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
    """Ghép các đoạn với silence (NGẮN HƠN)"""
    if len(segments) == 0:
        return np.array([], dtype=np.float32)
    result = [segments[0]]
    for i in range(1, len(segments)):
        silence_dur = silence_durations[min(i-1, len(silence_durations)-1)]
        silence = np.zeros(int(silence_dur * 24000), dtype=np.float32)
        result.extend([silence, segments[i]])
    return np.concatenate(result)

def speed_up_audio(wave, sample_rate, speed_factor=1.0):
    """
    Tăng tốc audio bằng cách resample
    speed_factor > 1.0: nhanh hơn
    speed_factor < 1.0: chậm hơn
    """
    if speed_factor == 1.0:
        return wave
    
    from scipy import signal
    
    # Tính số mẫu mới
    num_samples = int(len(wave) / speed_factor)
    
    # Resample
    wave_resampled = signal.resample(wave, num_samples)
    
    return wave_resampled.astype(np.float32)


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
        print(f"Starting Fast TTS (speed={speed})")
        print(f"{'='*60}")

        # 1. Thêm pause marker (NGẮN HƠN)
        text_with_markers = add_pause_markers(gen_text, pause_level)

        # 2. Chuẩn hóa
        normalized_text = post_process_text(TTSnorm(text_with_markers)).lower()

        # 3. Preprocess ref
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        print(f"Ref text: {ref_text[:100]}...")

        # 4. Sinh audio toàn bộ với speed parameter
        wave, sr, spectrogram = infer_process(
            ref_audio, ref_text.lower(), normalized_text, model, vocoder, 
            speed=speed  # Truyền speed vào model
        )

        print(f"Generated wave duration: {len(wave)/sr:.2f}s (before speedup)")

        # 5. Tách audio tại marker (nhanh hơn: 0.05s thay vì 0.08s)
        marker_segments = split_audio_by_marker(wave, sr, marker_duration_sec=0.05, energy_threshold=1e-5)
        print(f"Detected {len(marker_segments)} audio segments")

        # 6. Tính silence duration (GIẢM 50%)
        silence_map = {
            "Short":  [0.15, 0.05],   # Giảm từ 0.3s/0.15s
            "Medium": [0.25, 0.15],   # Giảm từ 0.5s/0.25s
            "Long":   [0.40, 0.25]    # Giảm từ 0.8s/0.4s
        }
        para_s, dia_s = silence_map.get(pause_level, [0.25, 0.15])

        # Đếm số câu
        sentences = re.findall(r'[.!?]+', gen_text)
        dialogue_sentences = len(re.findall(r'"[^"]*[.!?]', gen_text))
        narrative_sentences = len(sentences) - dialogue_sentences

        silence_durations = [para_s] * narrative_sentences + [dia_s] * dialogue_sentences
        silence_durations = silence_durations[:len(marker_segments)-1]

        if len(silence_durations) < len(marker_segments) - 1:
            silence_durations = [para_s] * (len(marker_segments) - 1)

        # 7. Ghép lại
        final_wave = insert_silence_between(marker_segments, silence_durations)

        # 8. TĂNG TỐC AUDIO (nếu speed > 1.0)
        if speed != 1.0:
            final_wave = speed_up_audio(final_wave, sr, speed_factor=speed)
            print(f"Applied speed factor: {speed}x")

        duration = len(final_wave) / sr
        print(f"Final audio: {duration:.2f}s | Speed: {speed}x")

        # 9. Lưu spectrogram
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
    # F5-TTS Fast: **Tốc độ đọc nhanh hơn**
    ### ✅ Giảm pause markers (|, ||, |||)
    ### ✅ Giảm 50% silence duration
    ### ✅ Hỗ trợ tăng tốc audio (speed > 1.0)
    """)

    with gr.Row():
        ref_audio = gr.Audio(label="Sample Voice", type="filepath")
        gen_text = gr.Textbox(
            label="Text to Generate", lines=10,
            placeholder="Hắn ngồi trên tàu. Mắt nhìn xa.\n\n\"Toa về bao lâu?\"\n\n\"Merci beaucoup!\""
        )

    with gr.Row():
        speed = gr.Slider(0.8, 2.0, 1.3, step=0.1, label="Speed (khuyên dùng 1.2-1.5)")
        pause_level = gr.Radio(["Short", "Medium", "Long"], "Short", label="Pause Level (dùng Short để nhanh)")

    btn = gr.Button("Generate Voice", variant="primary")
    with gr.Row():
        out_audio = gr.Audio(label="Generated Audio", type="numpy")
        out_spec = gr.Image(label="Spectrogram")

    gr.Markdown("""
    ### Cách tăng tốc độ đọc:
    1. **Tăng Speed slider**: 1.0 → 1.3 → 1.5 (nhanh hơn 30-50%)
    2. **Chọn Pause Level = Short**: Giảm khoảng lặng giữa các câu
    3. **Kết hợp cả 2**: Speed=1.5 + Short = nhanh nhất
    
    **Lưu ý:** Speed quá cao (>1.8) có thể làm giọng nghe không tự nhiên
    """)

    btn.click(infer_tts, [ref_audio, gen_text, speed, pause_level], [out_audio, out_spec])

demo.queue().launch(share=True)
