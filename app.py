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

def split_text_into_chunks(text, pause_paragraph_duration=0.8, pause_dialogue_duration=0.4):
    """
    Tách văn bản thành các câu riêng biệt và xác định thời gian dừng.
    
    Returns:
        list of tuples: [(sentence, pause_duration_in_seconds), ...]
    """
    lines = text.split('\n')
    chunks = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Kiểm tra nếu là hội thoại
        is_dialogue = line.startswith('"') or line.startswith('"') or line.startswith('"')
        pause_duration = pause_dialogue_duration if is_dialogue else pause_paragraph_duration
        
        # Tách câu dựa trên dấu câu
        sentences = re.split(r'([.!?]+)', line)
        
        current_sentence = ""
        for i, part in enumerate(sentences):
            if i % 2 == 0:  # Phần văn bản
                current_sentence += part
            else:  # Dấu câu
                current_sentence += part
                if current_sentence.strip():
                    chunks.append((current_sentence.strip(), pause_duration))
                current_sentence = ""
        
        # Xử lý phần còn lại (nếu có)
        if current_sentence.strip():
            chunks.append((current_sentence.strip(), pause_duration))
    
    return chunks

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
    """
    if not ref_audio_orig:
        raise gr.Error("Please upload a sample audio file.")
    if not gen_text.strip():
        raise gr.Error("Please enter the text content to generate voice.")
    
    try:
        # Cấu hình pause (giây)
        pause_configs = {
            "Short": (0.4, 0.2),    # Paragraph: 0.4s, Dialogue: 0.2s
            "Medium": (0.8, 0.4),   # Paragraph: 0.8s, Dialogue: 0.4s
            "Long": (1.2, 0.6)      # Paragraph: 1.2s, Dialogue: 0.6s
        }
        
        pause_paragraph, pause_dialogue = pause_configs.get(pause_level, (0.8, 0.4))
        
        print(f"\n🎛️ Pause config: Paragraph={pause_paragraph}s, Dialogue={pause_dialogue}s")
        
        # Tách văn bản thành các câu với thời gian dừng
        chunks = split_text_into_chunks(gen_text, pause_paragraph, pause_dialogue)
        
        print(f"\n📝 Total chunks: {len(chunks)}")
        
        # Preprocess reference audio
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        # Tạo audio cho từng câu và ghép lại
        audio_segments = []
        sample_rate = 24000
        
        for i, (sentence, pause_duration) in enumerate(chunks):
            print(f"\n🔄 Processing chunk {i+1}/{len(chunks)}: {sentence[:50]}...")
            
            # Chuẩn hóa văn bản
            normalized_text = post_process(TTSnorm(sentence)).lower()
            
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
            
            # Thêm khoảng im lặng (trừ câu cuối)
            if i < len(chunks) - 1:
                silence = create_silence(pause_duration, sample_rate)
                audio_segments.append(silence)
                print(f"   ⏸️  Added {pause_duration}s silence")
        
        # Ghép tất cả audio lại
        final_wave = np.concatenate(audio_segments)
        
        print(f"\n✅ Final audio length: {len(final_wave)/sample_rate:.2f}s")
        
        # Tạo spectrogram từ audio cuối cùng
        # Note: Bạn có thể cần import thêm để tạo spectrogram từ waveform
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_spectrogram:
            spectrogram_path = tmp_spectrogram.name
            # Tạo spectrogram đơn giản (hoặc bỏ qua nếu không cần thiết)
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 4))
            plt.specgram(final_wave, Fs=sample_rate, cmap='viridis')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectrogram')
            plt.colorbar(format='%+2.0f dB')
            plt.tight_layout()
            plt.savefig(spectrogram_path)
            plt.close()

        print("✅ Audio generated successfully!")
        return (sample_rate, final_wave), spectrogram_path
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Error generating voice: {e}")

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
            placeholder="Enter text with paragraphs and dialogue...\n\nExample:\nHe walked slowly. The sun was setting.\n\n\"How are you?\" she asked.\n\n\"I'm fine,\" he replied.", 
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
    ### 💡 How Smart Pause Works:
    
    | Feature | Description |
    |---------|-------------|
    | **Sentence Splitting** | Each sentence is processed separately |
    | **Real Silence** | Actual silent gaps are inserted (no fake comma sounds!) |
    | **Paragraph Pauses** | Longer silence after descriptive sentences |
    | **Dialogue Pauses** | Shorter silence between conversation lines |
    | **Three Levels** | Short (0.4s/0.2s), Medium (0.8s/0.4s), Long (1.2s/0.6s) |
    
    ### 📖 Usage Tips:
    - **Short**: Best for fast-paced reading, news, announcements
    - **Medium**: Recommended for stories, articles, general content
    - **Long**: Ideal for audiobooks, dramatic readings, poetry
    - Use double line breaks to separate major sections
    - Put dialogue in quotes: `"Hello," he said.`
    
    ### 🎯 Example Input:
    ```
    The old man sat by the river. He watched the boats pass.
    
    "Beautiful day, isn't it?" asked a stranger.
    
    "Indeed it is," the old man replied with a smile.
    ```
    
    ### ⚠️ Note:
    Processing longer texts will take more time as each sentence is generated separately then combined.
    """)
    
    with gr.Accordion("❗ Model Limitations", open=False):
        gr.Markdown("""
        1. **Numbers & Special Characters**: May not handle dates, phone numbers well
        2. **Audio Quality**: Use clear reference audio with minimal background noise
        3. **Reference Text**: Auto-transcribed using Whisper (may have errors)
        4. **Long Text**: Processing time increases with text length (each sentence processed separately)
        5. **Foreign Words**: May not pronounce non-Vietnamese words correctly
        """)

    # Connect button to function
    btn_synthesize.click(
        infer_tts, 
        inputs=[ref_audio, gen_text, speed, pause_level], 
        outputs=[output_audio, output_spectrogram]
    )

# Launch with public link
demo.queue().launch(share=True)
