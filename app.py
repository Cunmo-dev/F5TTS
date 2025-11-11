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
    temp_group = []  # [(sentence, pause), ...]
    group_id = 0
    temp_pause = pause_paragraph_duration
    
    for i, (sentence, pause, _) in enumerate(chunks):
        word_count = len(sentence.split())
        is_last = (i == len(chunks) - 1)
        
        if word_count >= 5:
            # Câu dài: xuất group trước (nếu có), rồi thêm câu này
            if temp_group:
                # Xuất group các câu ngắn
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
                # Xuất group
                for s, p in temp_group:
                    processed_chunks.append((s, p, group_id))
                group_id += 1
                temp_group = []
    
    # Xử lý group còn sót (nếu có)
    if temp_group:
        for s, p in temp_group:
            processed_chunks.append((s, p, group_id))
    
    # Log thông tin
    print(f"\n📦 Grouped sentences:")
    current_group = None
    for sentence, pause, gid in processed_chunks[:10]:
        if gid != current_group:
            print(f"\n   Group {gid}:")
            current_group = gid
        print(f"      - [{len(sentence.split())}w, {pause}s] {sentence[:60]}...")
    
    return processed_chunks


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
            raise gr.Error("No valid sentences found.")
        
        ref_audio, ref_text = preprocess_ref_audio_text(ref_audio_orig, "")
        
        audio_segments = []
        sample_rate = 24000
        
        # Xử lý từng câu
        current_group = None
        for i, (sentence, pause_duration, group_id) in enumerate(chunks):
            is_new_group = (group_id != current_group)
            is_last_in_group = (i == len(chunks) - 1 or chunks[i+1][2] != group_id)
            
            print(f"\n🔄 [{i+1}/{len(chunks)}] Group {group_id}: {sentence[:60]}...")
            
            # Chuẩn hóa
            normalized_text = post_process(safe_normalize(sentence))
            normalized_text = validate_text_for_tts(normalized_text)
            
            word_count = len(normalized_text.strip().split())
            if word_count < 2:
                print(f"   ⏭️ Skipped: too short ({word_count} words)")
                continue
            
            print(f"   📝 Normalized ({word_count} words): {normalized_text[:60]}...")
            
            # Generate audio với retry
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
                        print(f"   ❌ Max retries reached")
                        # Fallback: thử với 3 từ đầu
                        if len(normalized_text.split()) > 3:
                            print(f"   🔧 Trying first 3 words...")
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
                                print(f"   ✅ Simplified success")
                                success = True
                            except:
                                print(f"   ❌ Simplified also failed")
                        break
                    
                    import time
                    time.sleep(0.5)
            
            current_group = group_id
        
        if not audio_segments:
            raise gr.Error("No audio generated.")
            
        final_wave = np.concatenate(audio_segments)
        
        # Tính số group
        num_groups = len(set(gid for _, _, gid in chunks))
        print(f"\n✅ Final: {len(final_wave)/sample_rate:.2f}s from {len(chunks)} sentences in {num_groups} groups")
        
        # Spectrogram
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
        raise gr.Error(f"Error: {str(e)}")
