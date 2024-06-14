from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tensorflow as tf
import numpy as np
import wave
#import sounddevice as sd
import librosa
import scipy.io.wavfile as wavfile
import noisereduce as nr

app = Flask(__name__)
CORS(app)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

# #Function to record audio
# def record_audio(filename, duration=2, sample_rate=16000, channels=2, chunk_size=3200):
#     # Initialize pyaudio
#     p = pyaudio.PyAudio()

#     # Open stream
#     stream = p.open(format=pyaudio.paInt16,
#                     channels=channels,
#                     rate=sample_rate,
#                     input=True,
#                     frames_per_buffer=chunk_size)
#     print("Recording...")

#     frames = []
#     # Record for the given duration
#     for _ in range(0, int(sample_rate / chunk_size * duration)):
#         data = stream.read(chunk_size)
#         frames.append(data)

#     print("Recording finished.")
#     # Stop and close the stream
#     stream.stop_stream()
#     stream.close()
#     p.terminate()

#     # Convert stereo to mono if necessary
#     if channels == 2:
#         mono_frames = []
#         for frame in frames:
#             # Convert bytes to numpy array
#             stereo_data = np.frombuffer(frame, dtype=np.int16)
#             # Reshape array to 2D array (2 channels)
#             stereo_data = stereo_data.reshape(-1, 2)
#             # Average the two channels to get mono data
#             mono_data = stereo_data.mean(axis=1).astype(np.int16)
#             # Convert numpy array back to bytes
#             mono_frames.append(mono_data.tobytes())

#         # Replace frames with mono frames
#         frames = mono_frames

#     # Save the recorded data as a WAV file
#     with wave.open(filename, 'wb') as wf:
#         wf.setnchannels(1)  # Set number of channels to 1 (mono)
#         wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
#         wf.setframerate(sample_rate)
#         wf.writeframes(b''.join(frames))
     
        
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
  # Convert the waveform to a spectrogram via a STFT.
  spectrogram = tf.signal.stft(
      waveform, frame_length=255, frame_step=128)
  # Obtain the magnitude of the STFT.
  spectrogram = tf.abs(spectrogram)
  # Add a `channels` dimension, so that the spectrogram can be used
  # as image-like input data with convolution layers (which expect
  # shape (`batch_size`, `height`, `width`, `channels`).
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram      

#Commands 
commands =   ['aini', 'aiwa', 'aizi' ,'akaunzi', 'amai' ,'amaiguru' ,'amainini' ,'ambuka',
 'amburenzi', 'ambuuya' ,'ambuya' ,'ambuyamudendere' ,'amwa' ,'angere' ,'ani',
 'anika' ,'apa' ,'apo' ,'apuro' ,'asi' ,'ba' ,'baara' ,'baba' ,'babamudiki',
 'babamukuru' ,'badarika' ,'bado' ,'badza' ,'baka', 'bakatwa', 'bakayava' ,'bako',
 'bakwa' ,'bamba' ,'bambo' ,'bambomukunda' ,'banda' ,'bandakadzi' ,'bandamba',
 'bandana' ,'bandaoko' ,'bande' ,'bandika' ,'bandiko' ,'bandwe' ,'banga',
 'banganuka' ,'banganwa' ,'bangara' ,'bango' ,'banha' ,'bani' ,'banya' ,'banzuka',
 'bapatyuro' ,'bapiro' ,'bapu' ,'bara' ,'baramhanya' ,'barika' ,'barwe' ,'basa',
 'batapata' ,'batidza' ,'batsira' ,'baya' ,'bedura' ,'bereka' ,'besu',
 'bhabharasi' ,'bhadhara' ,'bhaibheri' ,'bhakiti' ,'bhanditi' ,'bhangi',
 'bhangu' ,'bhasera' ,'bhatiri' ,'bhavhu' ,'bhawa' ,'bhazi' ,'bhemba' ,'bhenda',
 'bhenji' ,'bheuka' ,'bhinhi' ,'bhinya' ,'bhinzi' ,'bhiridha' ,'bhiriji',
 'bhitiruti' ,'bhiya' ,'bhiza' ,'bhizautare' ,'bhodho' ,'bhora' ,'bhoso',
 'bhucha' ,'bhudhi' ,'bhurauzi' ,'bhuru' ,'bhutsu' ,'bibiritsa' ,'bofu' ,'boira',
 'bonde' ,'bonga' ,'bongozozo' ,'buda' ,'buruuru' ,'bveni' ,'chabudza',
 'chachura' ,'chaizvo' ,'chakata' ,'chakwaira' ,'chamhembe' ,'chamudzungururu',
 'chamunyurududu' ,'chamupupuri' ,'chando' ,'changamire' ,'chanza' ,'chanzi',
 'chapungu' ,'chapupu' ,'chapwititi' ,'chari' ,'chata' ,'chaya' ,'chayisa',
 'checheni' ,'chechetere' ,'chechi' ,'cheka' ,'chema' ,'chembere' ,'chenama',
 'cheneruka' ,'cheni' ,'chenjera' ,'chenura' ,'chera' ,'chete' ,'chibage',
 'chibayiro' ,'chibhende' ,'chibhubhubhu' ,'chibhunu' ,'chibodzwa' ,'chidembo',
 'chidhakwa' ,'chidhambakura' ,'chidhanana' ,'chidhinha' ,'chidhokwani',
 'chidhoma' ,'chidimburiso' ,'chifananidzo' ,'da' ,'dacha' ,'dada' ,'dadamira',
 'dafi' ,'daidza' ,'dakara' ,'dama' ,'damba' ,'dambudziko' ,'dambudzo' ,'dambuka',
 'dana' ,'danda' ,'dandadzi' ,'dandaura' ,'danga' ,'dare' ,'dede' ,'dedera',
 'dehenya', 'dekara' ,'deuka' ,'dhadha' ,'dhafu' ,'dhaka' ,'dhunduru' ,'dhura',
 'diki' ,'dikita' ,'dimbwa' ,'dimikira' ,'dimura' ,'dongorera' ,'donha',
 'donongora' ,'donzva' ,'dope' ,'dora' ,'dore' ,'doro' ,'dyara' ,'dyunga',
 'dzidza' ,'dzihwa' ,'dzimba' ,'dzimbo' ,'dzimira' ,'dzimura' ,'dzimwaira',
 'dzinga', 'dzinza' ,'dzipa' ,'dziva' ,'dziviriira' ,'dzivirira' ,'dzivisa',
 'dziya' ,'dzoira' ,'dzoka' ,'dzokorodza' ,"dzoran'ombe" ,'dzoro' ,'dzosera',
 'dzupuka' ,'dzvamuka' ,'dzvi' ,'dzviti' ,'dzvoti' ,'dzvova' ,'dzvura' ,'e',
 'edza' ,'edzaa' ,'edzesera' ,'eka' ,'ema' ,'embera' ,'enda' ,'enzana' ,'enzera',
 'enzvo' ,'era' ,'eredza' ,'ereka' ,'erekana' ,'erera' ,'ererana' ,'evhangero',
 'evo' ,'fa' ,'fachuka' ,'fadza' ,'faera' ,'fafadza' ,'fafitera' ,'fakaza',
 'fakazi' ,'famba' ,'fana' ,'fanana' ,'fananidza' ,'fani' ,'fanira' ,'fano',
 'fanza' ,'fara' ,'faranuka' ,'fararika' ,'fararira' ,'farariraa' ,'farira',
 'fashafasha' ,'fashama' ,'fashanuka' ,'fashuka' ,'fashura' ,'fasitera' ,'fata',
 'feira' ,'fekitari' ,'fema' ,'femba' ,'fembera' ,'femereka' ,'fenda' ,'ferefeta',
 'fetiraiza' ,'feya' ,'fikura' ,'fimbi' ,'finha' ,'finyama' ,'finyana' ,'fobha',
 'fodya' ,'fomoka' ,'fondodza' ,'foni' ,'foro' ,'foroma' ,'foromani', 'foto',
 'fototo' ,'fototoka' ,'foya' ,'fuchira' ,'fuduguka' ,'fudza' ,'fudzi' ,'fuga',
 'fugura' ,'fukatira' ,'fuko' ,'fuma' ,'fumuka' ,'funa' ,'funda' ,'fundo' ,'funga',
 'fungidziro' ,'funidza' ,'fupa' ,'fupika' ,'furaimachina' ,'furamera',
 'furidza' ,'furira' ,'furo' ,'furuka' ,'furura' ,'furusa' ,'fusha' ,'fushuka',
 'futa' ,'futada' ,'futi' ,'futunuka' ,'fuza']


# Define the directory you want to clear
directory_to_clear = "uploads"

output_file = 'uploads/recording.wav'
duration = 2  # Duration of recording in seconds

# Loading the model
loaded_model = tf.saved_model.load("trained_model")

# Route to serve the main HTML PAGE
@app.route('/')
def index():
    return render_template('index.html')

# Route to record
# @app.route("/record", methods=["POST"])
# def record_to_search():
#     # clear_directory(directory_to_clear)
#    # Records audio and  change it to wave, and channel to 1(mono)
#     record_audio(output_file, duration, channels=2)  # Record with stereo input
#     print(f"Audio recorded and saved as {output_file}")
    
#     # Clean(remove noise, normalise) the recorded saved file, and save the new cleaned file
#     input_file = output_file
#     sample_file = 'static/audio/output_cleaned_normalized_trimmed.wav'
#     process_audio(input_file, sample_file)
#     return jsonify({"message":"Audio recorded and saved"})

@app.route('/upload', methods=['POST'])
def upload():
    if 'audio' not in request.files:
        return 'No file part', 400
    file = request.files['audio']
    if file.filename == '':
        return 'No selected file', 400
    file.save(os.path.join(UPLOAD_FOLDER, 'recording.wav'))
    
    #Clean(remove noise, normalize) the recoded saved file, and save the new cleaned file
    input_file = output_file
    sample_file = 'uploads/output_cleaned_normalized_trimmed.wav'
    process_audio(input_file,sample_file)
    return 'File uploaded and processed successfully', 200


# Route to get transcription
@app.route("/transcribe", methods=["GET"])
def predict():
    file_final = 'uploads/output_cleaned_normalized_trimmed.wav'
    file_final =  tf.io.read_file(str(file_final))
    file_final, sample_rate = tf.audio.decode_wav(file_final, desired_channels=1, desired_samples= 16000 )
    file_final = tf.squeeze(file_final, axis=-1)
    waveform = file_final
    file_final = get_spectrogram(file_final)
    file_final = file_final[tf.newaxis, ...]
    
    prediction = loaded_model(file_final)
    
    #transcribing
    transcript = np.argmax(prediction, axis=1)
    print( f'transcript "{commands[transcript[0]]}"')
    return jsonify({"transcript": commands[transcript[0]]})

if __name__ == "__main__":
    app.run( debug=True,host='0.0.0.0', port=5000)