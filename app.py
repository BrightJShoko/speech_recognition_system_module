from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
import wave
import speech_recognition as sr
import pyaudio
#import sounddevice as sd
import librosa
import scipy.io.wavfile as wavfile
import noisereduce as nr

app = Flask(__name__)
CORS(app)

# Function to clear the directory
def clear_directory(directory):
    """
    Clears the contents of the specified directory.
    
    Args:
        directory (str): The path to the directory to be cleared.
        
    Returns:
        None
    """
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clear_directory(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")

#Function to record audio
def record_audio(filename, duration=2, sample_rate=16000, channels=2, chunk_size=3200):
    # Initialize pyaudio
    p = pyaudio.PyAudio()

    # Open stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=channels,
                    rate=sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size)
    print("Recording...")

    frames = []
    # Record for the given duration
    for _ in range(0, int(sample_rate / chunk_size * duration)):
        data = stream.read(chunk_size)
        frames.append(data)

    print("Recording finished.")
    # Stop and close the stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Convert stereo to mono if necessary
    if channels == 2:
        mono_frames = []
        for frame in frames:
            # Convert bytes to numpy array
            stereo_data = np.frombuffer(frame, dtype=np.int16)
            # Reshape array to 2D array (2 channels)
            stereo_data = stereo_data.reshape(-1, 2)
            # Average the two channels to get mono data
            mono_data = stereo_data.mean(axis=1).astype(np.int16)
            # Convert numpy array back to bytes
            mono_frames.append(mono_data.tobytes())

        # Replace frames with mono frames
        frames = mono_frames

    # Save the recorded data as a WAV file
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)  # Set number of channels to 1 (mono)
        wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(sample_rate)
        wf.writeframes(b''.join(frames))
     
        
#Functions to pre-process the audio
def load_audio(file_path, sr=16000):
    """ Load audio file with a fixed sample rate """
    y, sr = librosa.load(file_path, sr=sr)
    return y, sr

def save_audio(file_path, y, sr):
    """ Save audio file """
    wavfile.write(file_path, sr, (y * 32767).astype(np.int16))

def remove_noise(y, sr):
    """ Remove noise from audio signal """
    # Reducing noise
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    return y_denoised

def normalize_audio(y):
    """ Normalize audio signal to -1 to 1 """
    return y / np.max(np.abs(y))

def trim_silence(y, top_db=20):
    """ Trim leading and trailing silence from an audio signal """
    y_trimmed, index = librosa.effects.trim(y, top_db=top_db)
    return y_trimmed

def process_audio(input_file, output_file, sr=16000):
    """ Load, clean, normalize, trim, and save audio file with a fixed sample rate """
    # Loading the audio file
    y, sr = load_audio(input_file, sr)
    
    # Removing noise
    y_denoised = remove_noise(y, sr)
    
    # Normalizing the audio
    y_normalized = normalize_audio(y_denoised)
    
    # Trimming leading and trailing silence
    
    y_trimmed = trim_silence(y_normalized)
    
    # Saving the cleaned, normalized, and trimmed audio to a new file
    save_audio(output_file, y_trimmed, sr)
    
    print(f"Cleaned, normalized, and trimmed audio saved to {output_file}")
        
#Function to get the spectogram    
def get_spectrogram(waveform):
  # Zero-padding for waveforms with less than 16,000 samples
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
  #cast the waveform tensors dtype to float32
  waveform = tf.cast(waveform, dtype = tf.float32)  
  #concatinating to achieve the same length for all audio clips
  equal_length= tf.concat([waveform, zero_padding], 0)
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram          

#Commands 
commands =  ['dzosera', 'fachuka', 'fodya', 'asi', 'finyana', 'badza', 'dzinga', 'furura', 'dziva', 
             'bhabharasi', 'chenura', 'futada', 'dzoro', 'changamire', 'bhinhi', 'farira', 'dhaka', 
             'edzesera', 'bambo', 'bhanditi', 'bhiya', 'bandaoko', 'dambudziko', 'dimikira', 'dakara',
             'bhangu', 'aiwa', 'dzvura', 'barika', 'dzviti', 'fembera', 'funidza', 'fupa', 'fuchira', 
             'bandamba', 'fafadza', 'dana', 'dope', 'foromani', 'era', 'bakatwa', 'fetiraiza', 'dekara',
             'banganwa', 'foya', 'fakaza', 'chakwaira', 'chakata', 'amaiguru', 'dzinza', 'fudza', 'bhora', 
             'bhinzi', 'fusha', 'bande', 'amai', 'ani', 'finyama', 'anika', 'enzvo', 'fa', 'futa', 'chibhunu', 
             'dongorera', 'chera', 'fashanuka', 'fikura', 'chenama', 'baba', 'chapwititi', 'bonga', 'buruuru',
             'bhavhu', 'akaunzi', 'ema', 'bhurauzi', 'batsira', 'faranuka', 'bhaibheri', 'banzuka', 'bhemba',
             'fashama', 'fata', 'furaimachina', 'dhura', 'dandaura', 'angere', 'bakayava', 'danga', 'chenjera',
             'cheneruka', 'dzidza', 'bereka', 'fema', 'bibiritsa', 'bhiridha', 'chanza', 'funda', 'banga', 
             'famba', 'dhadha', 'edza', 'chidhoma', 'fuko', 'basa', 'fanira', 'chaizvo', 'dandadzi', 'ambuya',
             'fadza', 'bara', 'apa', 'dehenya', 'chapupu', 'futunuka', 'erera', 'fana', 'fashafasha', 'chibage',
             'aini', 'dzimwaira', 'e', 'fuga', 'chifananidzo', 'bapiro', 'fumuka', 'apo', 'bhudhi', 'besu', 
             'fupika', 'fararika', 'bhiza', 'chidimburiso', 'fanana', 'bhitiruti', 'amainini', 'ambuka', 
             'fushuka', 'dzivirira', 'banya', 'furira', 'feira', 'dama', 'baka', 'daidza', 'fekitari', 
             "dzoran'ombe", 'dimura', 'eredza', 'dzvova', 'chechi', 'ambuyamudendere', 'bamba', 'bheuka',
             'fara', 'bambomukunda', 'funa', 'dzvi', 'farariraa', 'foroma', 'fimbi', 'fakazi', 'fararira', 
             'chibodzwa', 'bandakadzi', 'amburenzi', 'ererana', 'bhatiri', 'bakwa', 'donha', 'dzvoti', 
             'chabudza', 'dedera', 'dzimbo', 'banha', 'furidza', 'chamhembe', 'dziya', 'dhunduru', 
             'chidhokwani', 'bandiko', 'dzvamuka', 'dzivisa', 'damba', 'buda', 'batidza', 'checheni', 'bhenji',
             'fafitera', 'ferefeta', 'dzoira', 'bangara', 'baya', 'foni', 'fototo', 'dora', 'fugura', 
             'fashuka', 'chibhende', 'bhadhara', 'donongora', 'babamukuru', 'bveni', 'finha', 'dada', 'funga',
             'barwe', 'dambuka', 'bhinya', 'cheni', 'chema', 'chidembo', 'bhakiti', 'cheka', 'bhutsu', 
             'fomoka', 'ereka', 'apuro', 'chachura', 'foto', 'foro', 'chamupupuri', 'chembere', 'bongozozo', 
             'chidhambakura', 'bonde', 'futi', 'bani', 'dzokorodza', 'fenda', 'bandika', 'dzimba', 'bhawa', 
             'bhasera', 'ba', 'chaya', 'diki', 'fuduguka', 'bedura', 'chechetere', 'fani', 'chibhubhubhu', 
             'fashura', 'bango', 'bapu', 'enda', 'fundo', 'dikita', 'chete', 'chidhakwa', 'dyunga', 'fukatira',
             'bhangi', 'enzera', 'bhenda', 'furuka', 'ambuuya', 'banda', 'dyara', 'chamudzungururu', 
             'femereka', 'furusa', 'chanzi', 'dhafu', 'chapungu', 'erekana', 'danda', 'embera', 'deuka', 
             'dimbwa', 'babamudiki', 'faera', 'bhucha', 'dafi', 'amwa', 'furo', 'dzihwa', 'dziviriira', 'doro',
             'fondodza', 'dzupuka', 'femba', 'chata', 'bhazi', 'dore', 'bapatyuro', 'fano', 'fasitera', 
             'chari', 'dzimira', 'enzana', 'chayisa', 'bhuru', 'fanza', 'fuza', 'dacha', 'baara', 'chando',
             'bako', 'eka', 'bofu', 'aizi', 'bhiriji', 'fuma', 'furamera', 'edzaa', 'badarika', 'feya', 
             'donzva', 'bhoso', 'dzoka', 'da', 'dadamira', 'batapata', 'chidhanana', 'dede', 'chidhinha', 
             'bandwe', 'boira', 'fototoka', 'fananidza', 'bado', 'dambudzo', 'banganuka', 'chibayiro', 
             'evhangero', 'dare', 'bhodho', 'bhizautare', 'fobha', 'fungidziro', 'dzimura', 'fudzi', 'evo',
             'bandana', 'dzipa', 'baramhanya', 'chamunyurududu']

# Define the directory you want to clear
directory_to_clear = "static/audio"

output_file = 'static/audio/output.wav'
duration = 2  # Duration of recording in seconds

# Loading the model
loaded_model = tf.saved_model.load("savd_modl")


# Route to serve the main HTML PAGE
@app.route('/')
def index():
    audio_file_path = "static/audio/output_cleaned_normalized_trimmed.wav" 
    return render_template('index.html', audio_file_path=audio_file_path)

# Route to record
@app.route("/record", methods=["POST"])
def record_to_search():
    clear_directory(directory_to_clear)
   # Records audio and  change it to wave, and channel to 1(mono)
    record_audio(output_file, duration, channels=2)  # Record with stereo input
    print(f"Audio recorded and saved as {output_file}")
    
    # Clean(remove noise, normalise) the recorded saved file, and save the new cleaned file
    input_file = output_file
    sample_file = 'static/audio/output_cleaned_normalized_trimmed.wav'
    process_audio(input_file, sample_file)
    return jsonify({"message":"Audio recorded and saved"})


# Route to get transcription
@app.route("/predict", methods=["GET"])
def predict():
    file_final = 'static/audio/output_cleaned_normalized_trimmed.wav'
    obj = wave.open(str( file_final), 'rb')
    n_samples = obj.getnframes()
    signal_wave = obj.readframes(n_samples)
    signal_array = np.frombuffer(signal_wave, dtype=np.int16)
    obj.close()
    print(signal_array.shape)

    waveform = signal_array/32768
    waveform = tf.convert_to_tensor(waveform, dtype=tf.float32)
    spec = get_spectrogram(waveform)
    spec = tf.expand_dims(spec, 0)
    prediction = loaded_model(spec)

    #transcribing
    transcript = np.argmax(prediction, axis= 1)
    print( f'Predicted transcript "{commands[transcript[0]]}"')
    return jsonify({"transcript": commands[transcript[0]]})

if __name__ == "__main__":
    app.run( debug=True,host='0.0.0.0', port=5000)