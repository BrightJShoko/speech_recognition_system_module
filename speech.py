import speech_recognition as sr

def record_audio(filename, duration=2, sample_rate=16000):
    recognizer = sr.Recognizer()
    microphone = sr.Microphone(sample_rate=sample_rate)
    
    print("Recording...")
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.record(source, duration=duration)
    print("Recording finished.")
    
    # Save the recorded data as a WAV file
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())

# Example usage:
record_audio("recorded_audio.wav", duration=2, sample_rate=16000)
