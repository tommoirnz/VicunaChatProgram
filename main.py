# Vicuna Chat Program using Speech recognition and text to speech output.
# Import operating system interface library
import os
# Import system-specific parameters and functions
import sys
# Import threading support for concurrent execution
import threading
# Import queue for thread-safe FIFO implementation
import queue
# Import tkinter for GUI creation
import tkinter as tk
# Import additional themed widgets and dialogs from tkinter
from tkinter import ttk, scrolledtext, filedialog, messagebox
# Import sounddevice for audio input/output operations
import sounddevice as sd
# Import numpy for numerical operations and array handling
import numpy as np
# Import speech_recognition for converting speech to text
import speech_recognition as sr
# Import torch (PyTorch) for deep learning model handling
import torch
# Import asyncio for asynchronous I/O operations
import asyncio
# Import edge_tts for Microsoft Edge Text-to-Speech
import edge_tts
# Import tempfile for temporary file creation
import tempfile
# Import AudioSegment from pydub for audio file handling
from pydub import AudioSegment
# Import simpleaudio for playing back audio
import simpleaudio as sa
# Import matplotlib.pyplot for plotting and saving images
import matplotlib.pyplot as plt
# Import io for in-memory byte streams
import io
# Import Image and ImageTk from PIL for image processing and Tkinter compatibility
from PIL import Image, ImageTk
# Import AutoTokenizer and AutoModelForCausalLM from transformers for language model setup
from transformers import AutoTokenizer, AutoModelForCausalLM
# Import ThreadPoolExecutor for managing a pool of threads
from concurrent.futures import ThreadPoolExecutor
# Import detect from langdetect for language detection functionality
from langdetect import detect  # Language detection library
# Import pyttsx3 for Windows SAPI5 text-to-speech functionality
import pyttsx3

# -------------------------------
# GLOBAL QUEUES

# Queue to hold transcription outputs from Google Speech Recognition
transcription_queue = queue.Queue()  # for Google transcript output
# Queue to hold responses from the Vicuna language model
vicuna_response_queue = queue.Queue()  # for Vicuna responses
# Queue to hold volume meter updates for audio input
volume_queue = queue.Queue()           # for volume meter updates

# Global variables for TTS playback
# Object for TTS playback control (e.g., simpleaudio play object)
tts_play_obj = None
# Last TTS audio generated stored as a tuple (raw_data, channels, sample_width, frame_rate)
last_tts_audio = None  # tuple: (raw_data, channels, sample_width, frame_rate)

# Global variables for controlling TTS systems
# Boolean flag to enable/disable TTS functionality
tts_enabled = True
# Default voice for Edge TTS if set
default_edge_voice = None
# Default voice for SAPI (Windows) TTS if set
default_sapi_voice = None

# Global variables for SAPI TTS (Windows)
sapi_engine = None  # The pyttsx3 engine instance for SAPI
sapi_thread = None  # Thread handling SAPI TTS playback
sapi_lock = threading.Lock()  # Lock to ensure only one SAPI thread runs at a time

# -------------------------------
# Language options for input

# List of supported languages with their locale codes for speech recognition
LANGUAGE_OPTIONS = [
    "English (US) - en-US",  # US English
    "English (UK) - en-GB",  # UK English
    "English (AU) - en-AU",  # Australian English
    "French (France) - fr-FR",  # French
    "German (Germany) - de-DE",  # German
    "Italian (Italy) - it-IT",  # Italian
    "Spanish (Spain) - es-ES",  # Spanish
    "Chinese (Mandarin) - zh-CN",  # Mandarin Chinese
    "Japanese - ja-JP",  # Japanese
    "Russian - ru-RU",  # Russian
    "Croatian - hr-HR",  # Croatian
    "Norwegian - nb-NO",  # Norwegian
    "Swedish - sv-SE",  # Swedish
    "Danish - da-DK",  # Danish
    "Irish - ga-IE",  # Irish
    "Welsh - cy-GB",  # Welsh
    "Afrikaans - af-ZA",  # Afrikaans
    "Thai - th-TH",  # Thai
    "Indonesian - id-ID",  # Indonesian
    "Hebrew - he-IL",  # Hebrew
    "Finnish - fi-FI",  # Finnish
    "Hindi - hi-IN",  # Hindi
    "Arabic - ar-SA",  # Arabic (Saudi Arabia)
    "Dutch - nl-NL",  # Dutch
    "Polish - pl-PL",  # Polish
    "Ukrainian - uk-UA",  # Ukrainian
    "Latin - la-LA",  # Latin
    "Scots Gaelic - gd-GB",  # Scots Gaelic
    "Greek (Greece) - el-GR"     # Added Greek
]

# -------------------------------
# Available Edge TTS Voices (include new languages)

# List of voices available for Microsoft Edge TTS; "null" disables Edge TTS
EDGE_VOICE_OPTIONS = ["null",  # "null" option disables Edge TTS
    "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
    "en-GB-LibbyNeural", "en-AU-NatashaNeural",
    "fr-FR-DeniseNeural", "fr-FR-HenriNeural",
    "de-DE-KatjaNeural", "de-DE-ConradNeural",
    "it-IT-IsabellaNeural", "it-IT-LuisaNeural",
    "es-ES-ElviraNeural", "es-ES-AlonsoNeural",
    "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
    "ja-JP-NanamiNeural", "ja-JP-KeitaNeural",
    "ru-RU-SvetlanaNeural",
    "hr-HR-RajaNeural",
    "nb-NO-FinnNeural",
    "sv-SE-MattiasNeural",
    "da-DK-ChristelNeural",
    "ga-IE-ColmNeural",
    "cy-GB-AledNeural",
    "af-ZA-WillemNeural",
    "th-TH-NiwatNeural",
    "id-ID-ArdiNeural",
    "he-IL-AvriNeural", "he-IL-AmiraNeural",
    "fi-FI-NooraNeural",
    "hi-IN-SwaraNeural",
    "ar-EG-ShakirNeural",
    "nl-NL-ColetteNeural",
    "pl-PL-MarekNeural",
    "uk-UA-OstapNeural",
    "el-GR-AthinaNeural"  # Added Greek Edge voice
]

# Get available SAPI5 voices using pyttsx3.
def get_sapi_voice_options():
    # Initialize the pyttsx3 engine for SAPI5 TTS
    engine = pyttsx3.init()
    # Retrieve available voices from the engine
    voices = engine.getProperty('voices')
    # Prepend "null" option to disable SAPI5 TTS and return the list of voice names
    return ["null"] + [v.name for v in voices]

# List of available SAPI voice options
SAPI_VOICE_OPTIONS = get_sapi_voice_options()

# -------------------------------
# Helper: Render text as LaTeX image using matplotlib

def render_text_as_latex(text):
    # Create a matplotlib figure with a specific size
    fig = plt.figure(figsize=(8, 6))
    # Add text to the figure centered horizontally and vertically
    fig.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
    # Create an in-memory bytes buffer
    buf = io.BytesIO()
    # Save the figure to the buffer as a PNG image
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    # Close the figure to free resources
    plt.close(fig)
    # Move to the beginning of the buffer
    buf.seek(0)
    # Open the image from the buffer
    img = Image.open(buf)
    return img

# -------------------------------
# Helper: Given a language code, return a matching voice.

def get_default_voice_for_language(lang_code, voice_options):
    # Mapping for special cases where the language code needs to be adjusted
    special_mapping = {
        "no": "nb",    # Norwegian: map 'no' to 'nb'
        "he": "he-il",
        "da": "da-dk",
        "sv": "sv-se",
        "la": "it-",
        "gd": "ga-"
    }
    # Convert language code to lowercase for matching
    lang_code = lang_code.lower()
    # Get the mapped code if it exists; otherwise use the original code
    mapped_code = special_mapping.get(lang_code, lang_code)
    # Loop through the list of voice options
    for voice in voice_options:
        # If the voice starts with the mapped language code, return it
        if voice.lower().startswith(mapped_code):
            return voice
    # Return None if no matching voice is found
    return None

# -------------------------------
# Microsoft TTS via edge-tts and simpleaudio for playback with fallback

async def _speak_text_async(text, voice, output_file):
    # Create a communicator instance for Edge TTS with the provided text and voice
    communicator = edge_tts.Communicate(text, voice=voice)
    # Asynchronously save the generated speech to the specified output file
    await communicator.save(output_file)

def speak_text_global(text, output_device):
    """Selects the TTS system based on the radio button selection.
       If both voices are set to "null" no TTS is performed."""
    global tts_play_obj, last_tts_audio, sapi_thread, sapi_engine

    # Retrieve the selected Edge TTS voice from the chat window
    edge_voice = vicuna_chat_window.edge_voice_combo.get()
    # Retrieve the selected SAPI TTS voice from the chat window
    sapi_voice = vicuna_chat_window.sapi_voice_combo.get()
    # Get the chosen TTS system (edge or sapi) from the radio button selection
    chosen_system = vicuna_chat_window.tts_system_var.get()

    # Check if TTS is globally disabled
    if not tts_enabled:
        print("TTS is globally disabled.")
        return

    # If Edge TTS system is selected
    if chosen_system == "edge":
        # Check if Edge voice is disabled
        if edge_voice == "null":
            print("Edge TTS is disabled (voice set to null).")
            return

        # Function to attempt Edge TTS generation using a specific voice
        def try_tts(chosen_voice):
            # Create a temporary file for the generated MP3
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_filename = f.name
            try:
                # Run asynchronous TTS generation and save output to temporary file
                asyncio.run(_speak_text_async(text, chosen_voice, temp_filename))
            except Exception as e:
                print(f"Edge TTS generation error with {chosen_voice}: {e}")
                os.remove(temp_filename)
                return None
            # Check if the generated file is empty
            if os.path.getsize(temp_filename) == 0:
                print(f"Edge TTS error: Generated MP3 file is empty for {chosen_voice}.")
                os.remove(temp_filename)
                return None
            try:
                # Load the generated audio using pydub
                audio = AudioSegment.from_file(temp_filename, format="mp3")
            except Exception as e:
                print(f"Error loading generated audio for {chosen_voice}: {e}")
                os.remove(temp_filename)
                return None
            # Remove the temporary file after loading
            os.remove(temp_filename)
            return audio

        # Attempt to generate audio using the selected Edge voice
        audio = try_tts(edge_voice)
        if audio is None:
            print("Edge TTS failed; no fallback.")
            return
        try:
            # Save the last generated TTS audio for replaying later
            last_tts_audio = (audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate)
            # If an output device is specified, use sounddevice to play audio
            if output_device is not None:
                samples = np.array(audio.get_array_of_samples())
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels))
                sd.play(samples, samplerate=audio.frame_rate, device=output_device)
                tts_play_obj = None
            else:
                # Otherwise, use simpleaudio for playback
                tts_play_obj = sa.play_buffer(
                    audio.raw_data,
                    num_channels=audio.channels,
                    bytes_per_sample=audio.sample_width,
                    sample_rate=audio.frame_rate
                )
        except Exception as e:
            print("Error playing Edge TTS audio:", e)

    # If SAPI TTS system is selected
    elif chosen_system == "sapi":
        # Check if SAPI voice is disabled
        if sapi_voice == "null":
            print("SAPI TTS is disabled (voice set to null).")
            return

        # Ensure that only one SAPI thread is running at a time using a lock
        with sapi_lock:
            if sapi_thread and sapi_thread.is_alive():
                stop_sapi()
                sapi_thread.join()

            # Define the function that will run in a separate thread to speak text using SAPI TTS
            def speak_sapi_thread(text_to_speak, sapi_voice):
                global sapi_engine
                # Initialize the SAPI engine
                sapi_engine = pyttsx3.init()
                # Retrieve available voices
                voices = sapi_engine.getProperty('voices')
                # Loop through voices to find a match for the selected voice
                for v in voices:
                    if sapi_voice.lower() in v.name.lower():
                        sapi_engine.setProperty('voice', v.id)
                        break
                # Queue the text to be spoken
                sapi_engine.say(text_to_speak)
                try:
                    # Run the engine and wait until speaking is finished
                    sapi_engine.runAndWait()
                except RuntimeError as e:
                    print("RuntimeError in SAPI thread:", e)
                # Attempt to stop the engine without printing errors
                try:
                    sapi_engine.stop()
                except Exception:
                    pass
                # Clear the engine instance
                sapi_engine = None

            # Create and start a daemon thread for SAPI TTS
            sapi_thread = threading.Thread(
                target=speak_sapi_thread,
                args=(text, sapi_voice),
                daemon=True
            )
            sapi_thread.start()
    else:
        # If an unknown TTS system is selected, print an error message
        print("Unknown TTS system selected.")

# Function to stop SAPI TTS playback
def stop_sapi():
    global sapi_engine
    if sapi_engine is not None:
        try:
            # Attempt to stop the SAPI engine
            sapi_engine.stop()
        except Exception:
            pass  # Suppress any exception
        sapi_engine = None
        print("SAPI TTS stopped.")
    else:
        print("SAPI TTS engine already None.")

# Function to stop any ongoing TTS playback from either system
def stop_tts():
    global tts_play_obj
    try:
        # If using simpleaudio and playback is active, stop it
        if tts_play_obj is not None and tts_play_obj.is_playing():
            tts_play_obj.stop()
    except Exception:
        pass
    # Stop any playback from sounddevice
    sd.stop()
    tts_play_obj = None
    # Also stop SAPI TTS if it is running
    stop_sapi()

# Function to replay the last generated TTS audio
def replay_tts():
    global last_tts_audio, tts_play_obj
    output_device = None
    try:
        # Get the selected output device from the chat window
        selected_output = vicuna_chat_window.output_combo.get()
        output_device = int(selected_output.split(":")[0])
    except Exception:
        output_device = None
    if last_tts_audio is None:
        print("No TTS audio available to replay.")
        return
    try:
        # Replay the audio using simpleaudio
        tts_play_obj = sa.play_buffer(
            last_tts_audio[0],
            num_channels=last_tts_audio[1],
            bytes_per_sample=last_tts_audio[2],
            sample_rate=last_tts_audio[3]
        )
    except Exception as e:
        print("Error replaying TTS audio:", e)

# -------------------------------
# Global conversation history for Vicuna

# List to store conversation history for the Vicuna chat (system, user, AI messages)
vicuna_conversation_history = []

# -------------------------------
# VICUNA Chat Response Function (uses beam search Spinbox)

def vicuna_chat_response(user_text, temperature=0.9, top_p=0.9, rep_penalty=1.1):
    global vicuna_conversation_history, beam_search_spinbox
    # If conversation history is empty, initialize it with a system prompt
    if not vicuna_conversation_history:
        system_prompt = (
            "system: You are a helpful, friendly Scottish AI that responds in Aberdeen Doric English. "
            "You can speak many languages but choose English first unless I specify different. "
                    )
        vicuna_conversation_history.append(("system", system_prompt))
    # Append the user's text to the conversation history
    vicuna_conversation_history.append(("user", user_text))
    # Create a prompt string from the conversation history for the model
    prompt = "\n".join(f"{role}: {text}" for role, text in vicuna_conversation_history) + "\nAI:"
    # Tokenize the prompt and move input to the appropriate device
    input_ids = vicuna_tokenizer(prompt, return_tensors="pt").input_ids.to(vicuna_model.device)
    # Get the beam search parameter from the spinbox
    num_beams = int(beam_search_spinbox.get())
    with torch.no_grad():
        # Generate the model's response using the provided parameters
        output = vicuna_model.generate(
            input_ids,
            max_new_tokens=4096,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            num_beams=num_beams,
            do_sample=True
        )
    # Decode the generated tokens into text
    full_response = vicuna_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Extract the latest AI response from the generated text
    if "AI:" in full_response:
        latest_response = full_response.split("AI:")[-1].strip()
    else:
        latest_response = full_response.strip()
    # Append the AI response to the conversation history
    vicuna_conversation_history.append(("ai", latest_response))
    return latest_response

# -------------------------------
# Combined ASR Application (Main window with Google ASR)

class CombinedASRApp:
    def __init__(self, root):
        # Store the root Tkinter window reference
        self.root = root
        # Set the title of the main window
        self.root.title("Speech Recognition - English Only")
        # Set the background color of the window
        self.root.configure(bg="white")
        # Set the window geometry (width x height)
        self.root.geometry("400x700")
        # Bind Ctrl+Left-click to toggle recording and submitting audio
        self.root.bind_all("<Control-Button-1>", lambda event: self.toggle_record_and_submit())
        # Set the audio sample rate
        self.samplerate = 16000
        # Set the chunk size for audio processing
        self.chunk_size = 2048
        # Set the default microphone gain
        self.gain = 1.0
        # Initialize a list to store audio chunks
        self.buffered_chunks = []
        # Set the default spoken language code for recognition
        self.spoken_language_code = "en-US"
        # Tkinter variable to store buffer size (number of chunks)
        self.buffer_size_var = tk.IntVar(value=100)
        # Default overlap percentage for buffering audio
        self.overlap_percentage = 4
        # Create a thread pool for handling audio processing in the background
        self.executor = ThreadPoolExecutor(max_workers=2)
        # Flag to indicate whether recording is in progress
        self.is_recording = False
        # Build the GUI widgets for the application
        self.build_widgets()
        # Schedule volume queue processing to update the volume meter
        self.root.after(100, self.process_volume_queue)

    def build_widgets(self):
        # Create and pack a label for microphone device selection
        tk.Label(self.root, text="Select Microphone Device:", bg="white").pack(pady=5)
        # Create a read-only combobox for selecting the microphone device
        self.device_combobox = ttk.Combobox(self.root, state="readonly", width=50)
        self.device_combobox.pack(pady=5)
        # Populate the combobox with available audio devices
        self.list_audio_devices()

        # Create and pack a label for input language selection
        tk.Label(self.root, text="Select Input Language:", bg="white").pack(pady=5)
        # Create a read-only combobox for selecting the input language
        self.language_combobox = ttk.Combobox(self.root, state="readonly", width=50, values=LANGUAGE_OPTIONS)
        self.language_combobox.pack(pady=5)
        # Set the default language to the first option
        self.language_combobox.current(0)
        # Bind the selection event to update the spoken language code
        self.language_combobox.bind("<<ComboboxSelected>>", self.update_language)

        # Create and pack a label for microphone gain control
        tk.Label(self.root, text="Mic Gain:", bg="white").pack(pady=5)
        # Create a slider to control the microphone gain
        self.gain_slider = tk.Scale(self.root, from_=1.0, to=4.0, resolution=0.1,
                                    orient="horizontal", length=225, command=self.set_gain)
        self.gain_slider.set(1.0)
        self.gain_slider.pack(pady=5)

        # Create a frame for buffer size controls
        frame_buf = tk.Frame(self.root, bg="white")
        frame_buf.pack(pady=5)
        # Create and pack a label for buffer size selection
        tk.Label(frame_buf, text="Buffer Size (chunks):", bg="white").pack(side="left", padx=5)
        # Create a slider to control the buffer size and bind it to the buffer_size_var variable
        self.buffer_size_slider = tk.Scale(frame_buf, from_=20, to=140, resolution=10,
                                           orient="horizontal", variable=self.buffer_size_var, length=200)
        self.buffer_size_slider.pack(side="left", padx=5)

        # Create a frame for overlap percentage controls
        frame_overlap = tk.Frame(self.root, bg="white")
        frame_overlap.pack(pady=5)
        # Create and pack a label for overlap percentage
        tk.Label(frame_overlap, text="Overlap (%):", bg="white").pack(side="left", padx=5)
        # Create a slider to control the overlap percentage and bind its callback
        self.overlap_slider = tk.Scale(frame_overlap, from_=0, to=20, resolution=1,
                                       orient="horizontal", length=200, command=self.set_overlap)
        self.overlap_slider.set(self.overlap_percentage)
        self.overlap_slider.pack(side="left", padx=5)

        # Create and pack a label for the volume meter
        tk.Label(self.root, text="Volume:", bg="white").pack(pady=5)
        # Create a progress bar to display the audio volume level
        self.volume_bar = ttk.Progressbar(self.root, orient="horizontal",
                                          mode="determinate", maximum=100, length=375)
        self.volume_bar.pack(pady=5)

        # Create and pack a label for beam search selection
        tk.Label(self.root, text="Beam Search (1-9):", bg="white").pack(pady=5)
        # Declare the global beam_search_spinbox to use in the vicuna_chat_response function
        global beam_search_spinbox
        # Create a spinbox to select the number of beams for search in the model
        beam_search_spinbox = tk.Spinbox(self.root, from_=1, to=9, width=5)
        beam_search_spinbox.pack(pady=5)

        # Create a button to toggle recording and submitting audio
        self.toggle_button = tk.Button(self.root, text="Record & Submit",
                                       command=self.toggle_record_and_submit,
                                       font=("Helvetica", 14), bg="#4CAF50", fg="white", bd=3, relief="raised")
        self.toggle_button.pack(pady=10, fill="x", ipady=10)

        # Create a button to toggle TTS on/off
        self.tts_toggle_button = tk.Button(self.root, text="Disable TTS", command=self.toggle_tts,
                                       font=("Helvetica", 12), bg="#9C27B0", fg="white", bd=3, relief="raised")
        self.tts_toggle_button.pack(pady=5, fill="x", ipady=5)

        # Create a button to clear the transcript text
        self.clear_text_button = tk.Button(self.root, text="Clear Transcript Text",
                                           command=self.clear_transcript_text,
                                           font=("Helvetica", 12), bg="#F44336", fg="white", bd=3, relief="raised")
        self.clear_text_button.pack(pady=5, fill="x", ipady=5)

    def toggle_tts(self):
        global tts_enabled, default_edge_voice, default_sapi_voice
        # Toggle the TTS enabled state
        tts_enabled = not tts_enabled
        if tts_enabled:
            # If enabling TTS, update button text and restore default voices if available
            self.tts_toggle_button.config(text="Disable TTS")
            if default_edge_voice:
                vicuna_chat_window.edge_voice_combo.set(default_edge_voice)
            if default_sapi_voice:
                vicuna_chat_window.sapi_voice_combo.set(default_sapi_voice)
        else:
            # If disabling TTS, update button text and store current voice settings then disable them
            self.tts_toggle_button.config(text="Enable TTS")
            default_edge_voice = vicuna_chat_window.edge_voice_combo.get()
            default_sapi_voice = vicuna_chat_window.sapi_voice_combo.get()
            vicuna_chat_window.edge_voice_combo.set("null")
            vicuna_chat_window.sapi_voice_combo.set("null")

    def clear_transcript_text(self):
        # Clear the transcript text in the transcript window if it exists
        if transcript_window:
            transcript_window.text_widget.delete("1.0", tk.END)

    def update_language(self, event=None):
        # Update the spoken language code based on the selection in the language combobox
        selected = self.language_combobox.get()
        self.spoken_language_code = selected.split("-")[-1].strip()

    def list_audio_devices(self):
        # Query available audio devices using sounddevice
        devices = sd.query_devices()
        device_list = []
        # Loop through devices and add only those with input channels
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                device_list.append(f"{i}: {dev['name']}")
        # Set the values for the device combobox
        self.device_combobox['values'] = device_list
        # Set the first available device as default if any exist
        if device_list:
            self.device_combobox.current(0)

    def set_gain(self, value):
        try:
            # Update the gain value for microphone input
            self.gain = float(value)
        except Exception as e:
            print("Error setting gain:", e)

    def set_overlap(self, value):
        try:
            # Update the overlap percentage for audio buffering
            self.overlap_percentage = float(value)
        except Exception as e:
            print("Error setting overlap:", e)

    def process_volume_queue(self):
        try:
            # Process the volume queue and update the volume progress bar
            while not volume_queue.empty():
                vol = volume_queue.get_nowait()
                self.volume_bar['value'] = vol
        except queue.Empty:
            pass
        # Schedule the next volume update
        self.root.after(100, self.process_volume_queue)

    def audio_callback(self, indata, frames, time_info, status):
        try:
            # Print any audio stream status messages
            if status:
                print("Audio status:", status)
            # Apply the gain to the incoming audio data
            indata = indata * self.gain
            # Define a threshold below which the sound is considered silence
            silence_threshold = 0.02
            if np.linalg.norm(indata) < silence_threshold:
                return
            # Append a copy of the current audio chunk to the buffer
            self.buffered_chunks.append(indata.copy())
            # Calculate the volume for the current audio chunk and update the volume queue
            vol = np.linalg.norm(indata) * 10
            volume_queue.put(min(vol, 100))
            # Retrieve the current buffer size from the slider variable
            current_buf_size = self.buffer_size_var.get()
            # If the number of buffered chunks exceeds the buffer size, process the buffer
            if len(self.buffered_chunks) >= current_buf_size:
                self.executor.submit(self.worker_thread, self.buffered_chunks.copy())
                # Calculate how many chunks to retain based on the overlap percentage
                overlap = self.overlap_percentage / 100.0
                retain = int(overlap * len(self.buffered_chunks))
                # Retain the last 'retain' chunks if overlap is set
                self.buffered_chunks = self.buffered_chunks[-retain:] if retain > 0 else []
        except Exception as e:
            print("Error in audio callback:", e)

    def worker_thread(self, audio_chunks):
        # Process the audio buffer in a background thread
        self.process_audio_buffer(audio_chunks)

    def process_audio_buffer(self, audio_chunks):
        try:
            # Concatenate the audio chunks into a single array and convert to float32
            combined_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)
            # Initialize the speech recognizer
            recognizer = sr.Recognizer()
            # Convert the audio to int16 format for recognition
            audio_int16 = np.int16(combined_audio * 32767)
            # Convert the audio data to bytes
            audio_bytes = audio_int16.tobytes()
            # Create an AudioData instance for recognition
            audio_data = sr.AudioData(audio_bytes, self.samplerate, 2)
            # Recognize the speech using Google's speech recognition
            text = recognizer.recognize_google(audio_data, language=self.spoken_language_code)
            # If recognized text is not empty, put it into the transcription queue
            if text.strip():
                transcription_queue.put(text + "\n")
            return text
        except sr.UnknownValueError:
            # Return empty string if speech was unintelligible
            return ""
        except Exception as e:
            # Put an error message in the transcription queue if an error occurs
            transcription_queue.put("Error processing audio: " + str(e) + "\n")
            return ""

    def toggle_record_and_submit(self):
        if not self.is_recording:
            # If not recording, clear the audio buffer and start recording
            self.buffered_chunks = []
            try:
                # Get the microphone index from the device combobox
                mic_index = int(self.device_combobox.get().split(":")[0])
            except Exception as e:
                messagebox.showerror("Device Error", "No microphone device selected.")
                return
            try:
                # Create and start an InputStream for audio recording
                self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, device=mic_index,
                                             blocksize=self.chunk_size, callback=self.audio_callback)
                self.stream.start()
            except Exception as e:
                messagebox.showerror("Audio Stream Error", str(e))
                return
            # Set recording flag to True and update UI elements accordingly
            self.is_recording = True
            self.toggle_button.config(text="Stop & Submit")
            self.root.configure(bg="red")
        else:
            # If recording is in progress, stop the stream
            if hasattr(self, "stream"):
                self.stream.stop()
            # Update the recording flag and UI elements after stopping
            self.is_recording = False
            self.toggle_button.config(text="Record & Submit")
            self.root.configure(bg="white")
            # Process the buffered audio chunks after stopping recording
            transcript = self.process_audio_buffer(self.buffered_chunks)
            self.buffered_chunks = []
            # Insert the recognized text into the Vicuna chat window as user input
            vicuna_chat_window.insert_user_text(transcript)

# -------------------------------
# Transcript Window

class TranscriptWindow(tk.Toplevel):
    def __init__(self, master, queue):
        # Initialize a new top-level window for displaying transcripts
        super().__init__(master)
        self.title("Transcript")
        self.geometry("500x400")
        # Create a scrolled text widget for displaying the transcript
        self.text_widget = scrolledtext.ScrolledText(self, wrap="word", font=("Arial", 16))
        self.text_widget.pack(expand=True, fill="both")
        # Create a slider to adjust font size of the transcript text
        self.font_size_slider = ttk.Scale(self, from_=8, to=48, orient="horizontal", command=self.change_font_size)
        self.font_size_slider.set(16)
        self.font_size_slider.pack(fill="x")
        # Create a button to save the transcript to a file
        self.save_button = tk.Button(self, text="Save Transcript", command=self.save_transcript)
        self.save_button.pack(pady=5)
        # Store the transcription queue reference
        self.queue = queue
        # Start periodic updates of the transcript text
        self.after(100, self.update_text)

    def change_font_size(self, value):
        # Change the font size of the transcript text widget
        new_size = int(float(value))
        self.text_widget.config(font=("Arial", new_size))
        self.update_idletasks()

    def update_text(self):
        # Update the transcript text widget with messages from the transcription queue
        while not self.queue.empty():
            msg = self.queue.get()
            if msg == "__CLEAR__":
                self.text_widget.delete("1.0", tk.END)
            else:
                self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
        self.after(100, self.update_text)

    def save_transcript(self):
        # Open a file dialog to choose where to save the transcript
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            # Write the contents of the transcript text widget to the selected file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.text_widget.get("1.0", tk.END))

# -------------------------------
# VICUNA Chat Window

class VicunaChatWindow(tk.Toplevel):
    def __init__(self, master):
        # Initialize a new top-level window for the Vicuna chat
        super().__init__(master)
        self.title("Vicuna Chat")
        # Create a frame for top control buttons and pack it at the top
        top_controls_frame = tk.Frame(self)
        top_controls_frame.pack(side="top", fill="x", padx=10, pady=5)
        # Create a button to clear the chat display
        self.clear_chat_button = tk.Button(top_controls_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_chat_button.pack(side="left", padx=5)
        # Create a button to clear the conversation history
        self.clear_history_button = tk.Button(top_controls_frame, text="Clear History", command=self.clear_history)
        self.clear_history_button.pack(side="left", padx=5)
        # Create another frame for various option controls
        options_frame = tk.Frame(self)
        options_frame.pack(side="top", fill="x", padx=10, pady=5)
        # Create and position a label for Edge TTS voice selection
        tk.Label(options_frame, text="Edge TTS Voice:").grid(row=0, column=0, padx=5)
        # Create a combobox for selecting the Edge TTS voice
        self.edge_voice_combo = ttk.Combobox(options_frame, values=EDGE_VOICE_OPTIONS, state="readonly", width=20)
        self.edge_voice_combo.grid(row=0, column=1, padx=5)
        self.edge_voice_combo.current(1)
        # Create and position a label for SAPI5 TTS voice selection
        tk.Label(options_frame, text="SAPI5 Voice:").grid(row=0, column=2, padx=5)
        # Create a combobox for selecting the SAPI5 TTS voice
        self.sapi_voice_combo = ttk.Combobox(options_frame, values=SAPI_VOICE_OPTIONS, state="readonly", width=20)
        self.sapi_voice_combo.grid(row=0, column=3, padx=5)
        self.sapi_voice_combo.current(0)
        # Bind events to update voices if selection changes
        self.edge_voice_combo.bind("<<ComboboxSelected>>", self.on_edge_voice_change)
        self.sapi_voice_combo.bind("<<ComboboxSelected>>", self.on_sapi_voice_change)

        # Create and position a label for TTS system selection
        tk.Label(options_frame, text="TTS System:").grid(row=0, column=4, padx=5)
        # Create a StringVar to store the selected TTS system and default to "edge"
        self.tts_system_var = tk.StringVar(value="edge")
        # Create radio buttons to choose between Edge TTS and SAPI TTS
        tk.Radiobutton(options_frame, text="Edge TTS", variable=self.tts_system_var, value="edge").grid(row=0, column=5, padx=5)
        tk.Radiobutton(options_frame, text="SAPI TTS", variable=self.tts_system_var, value="sapi").grid(row=0, column=6, padx=5)

        # Create and position a label for output device selection
        tk.Label(options_frame, text="Output Device:").grid(row=0, column=7, padx=5)
        # Create a list of available output devices for audio playback
        output_devices = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]
        # Create a combobox to select the audio output device
        self.output_combo = ttk.Combobox(options_frame, values=output_devices, state="readonly", width=20)
        self.output_combo.grid(row=0, column=8, padx=5)
        if output_devices:
            self.output_combo.current(0)

        # Create a checkbutton to toggle automatic TTS playback of AI responses
        self.auto_tts_var = tk.BooleanVar(value=True)
        self.auto_tts_check = tk.Checkbutton(options_frame, text="Auto TTS", variable=self.auto_tts_var)
        self.auto_tts_check.grid(row=0, column=9, padx=5)
        # Create a button to stop TTS playback
        self.stop_tts_button = tk.Button(options_frame, text="Stop TTS", command=stop_tts)
        self.stop_tts_button.grid(row=0, column=10, padx=5)
        # Create a button to replay the last TTS audio
        self.replay_tts_button = tk.Button(options_frame, text="Replay TTS", command=replay_tts)
        self.replay_tts_button.grid(row=0, column=11, padx=5)

        # Create a scrolled text widget to display the chat conversation
        self.chat_display = scrolledtext.ScrolledText(self, wrap="word", font=("Arial", 12),
                                                      bg="black", fg="white", insertbackground="white")
        self.chat_display.pack(expand=True, fill="both")
        # Create a slider to adjust the font size of the chat display text
        self.font_size_slider = ttk.Scale(self, from_=8, to=48, orient="horizontal", command=self.change_text_size)
        self.font_size_slider.set(12)
        self.font_size_slider.pack(fill="x")
        # Create a button to save the chat transcript to a file
        self.save_chat_button = tk.Button(self, text="Save Chat Transcript", command=self.save_chat_transcript)
        self.save_chat_button.pack(pady=5)
        # Create a button to render the chat conversation as a LaTeX image
        self.render_chat_button = tk.Button(self, text="Render Chat as LaTeX", command=self.render_chat_as_latex,
                                            font=("Helvetica", 12), bg="#FF9800", fg="white", bd=3, relief="raised")
        self.render_chat_button.pack(pady=5, fill="x")
        # Create a frame for user input and submit button
        input_frame = tk.Frame(self)
        input_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        # Create a text widget for the user to input text
        self.user_input = tk.Text(input_frame, height=4, font=("Arial", 12))
        self.user_input.pack(fill="x", padx=10, pady=(0, 5))
        # Create a button to submit the user input
        self.submit_button = tk.Button(input_frame, text="Submit", command=self.on_submit)
        self.submit_button.pack(padx=10, pady=(0, 5))
        # Start periodic checking of the Vicuna response queue
        self.after(100, self.check_vicuna_response_queue)

    def on_edge_voice_change(self, event):
        # If both Edge and SAPI voices are set to "null", ensure at least one voice is selected
        if self.edge_voice_combo.get() == "null" and self.sapi_voice_combo.get() == "null":
            if len(SAPI_VOICE_OPTIONS) > 1:
                self.sapi_voice_combo.current(1)

    def on_sapi_voice_change(self, event):
        # If both SAPI and Edge voices are set to "null", ensure at least one voice is selected
        if self.sapi_voice_combo.get() == "null" and self.edge_voice_combo.get() == "null":
            if len(EDGE_VOICE_OPTIONS) > 1:
                self.edge_voice_combo.current(1)

    def clear_chat(self):
        # Clear the chat display widget
        self.chat_display.delete("1.0", tk.END)

    def clear_history(self):
        # Clear both the chat display and the conversation history
        self.clear_chat()
        global vicuna_conversation_history
        vicuna_conversation_history = []

    def render_chat_as_latex(self):
        # Get the entire chat conversation text
        chat_text = self.chat_display.get("1.0", tk.END)
        try:
            # Render the chat text as a LaTeX image using the helper function
            img = render_text_as_latex(chat_text)
            # Create a new top-level window to display the rendered image
            window = tk.Toplevel(self)
            window.title("Rendered Chat")
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(window, image=photo)
            label.image = photo  # Keep a reference to avoid garbage collection
            label.pack()
        except Exception as e:
            messagebox.showerror("Rendering Error", f"Error rendering chat as LaTeX: {e}")

    def change_text_size(self, value):
        # Update the font size of the chat display based on the slider value
        new_size = int(float(value))
        self.chat_display.config(font=("Arial", new_size))
        self.update_idletasks()

    def save_chat_transcript(self):
        # Open a file dialog to choose where to save the chat transcript
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            # Save the contents of the chat display widget to the selected file
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.chat_display.get("1.0", tk.END))

    def on_submit(self, event=None):
        # Get the user input text from the input widget and strip any extra whitespace
        text = self.user_input.get("1.0", tk.END).strip()
        if text:
            # Append the user's text to the chat display
            self.append_text("User: " + text + "\n")
            # Clear the user input widget
            self.user_input.delete("1.0", tk.END)
            # Process the user input in a separate thread to generate a response from Vicuna
            threading.Thread(target=self.process_vicuna_input, args=(text,), daemon=True).start()

    def process_vicuna_input(self, text):
        # Generate a response from the Vicuna language model for the given text
        response = vicuna_chat_response(text)
        # Put the response into the Vicuna response queue
        vicuna_response_queue.put(response)

    def check_vicuna_response_queue(self):
        try:
            # Process all responses in the Vicuna response queue
            while True:
                response = vicuna_response_queue.get_nowait()
                self.append_text("AI: " + response + "\n\n")
                try:
                    # Detect the language of the AI response
                    lang = detect(response)
                    # Get the default voice matching the detected language
                    default_voice = get_default_voice_for_language(lang, self.edge_voice_combo["values"])
                    if default_voice:
                        # Set the Edge voice combobox to the detected default voice
                        for idx, voice in enumerate(self.edge_voice_combo["values"]):
                            if voice == default_voice:
                                self.edge_voice_combo.current(idx)
                                break
                except Exception as e:
                    print("Language detection error:", e)
                # If auto TTS is enabled, start TTS playback in a separate thread
                if self.auto_tts_var.get():
                    selected_output = self.output_combo.get()
                    try:
                        output_device_index = int(selected_output.split(":")[0])
                    except Exception as e:
                        print("Error parsing output device:", e)
                        output_device_index = None
                    threading.Thread(target=speak_text_global, args=(response, output_device_index),
                                     daemon=True).start()
        except queue.Empty:
            pass
        # Schedule the next check of the response queue
        self.after(100, self.check_vicuna_response_queue)

    def append_text(self, text):
        # Append the given text to the chat display widget
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)

    def insert_user_text(self, text):
        # Insert user text into the input widget and trigger submission
        self.user_input.delete("1.0", tk.END)
        self.user_input.insert("1.0", text)
        self.on_submit()

# -------------------------------
# VICUNA MODEL SETUP

# Define the path to the Vicuna model (adjust path as needed)
vicuna_model_path = r"C:\models\vicuna-13b-v1.3"  # Adjust path as needed
print("Loading Vicuna Tokenizer...")
# Load the tokenizer for the Vicuna model from the specified path
vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_model_path, use_fast=False)
print("Loading Vicuna Model in 4-bit mode...")
# Load the Vicuna language model in 4-bit mode with specific quantization parameters
vicuna_model = AutoModelForCausalLM.from_pretrained(
    vicuna_model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=False,
    device_map="auto"
)
print("Vicuna Model Loaded Successfully!")

# -------------------------------
# MAIN EXECUTION

# If the script is run directly (not imported as a module)
if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()
    # Initialize the CombinedASRApp with the root window
    app = CombinedASRApp(root)
    # Create the TranscriptWindow for displaying audio transcriptions
    transcript_window = TranscriptWindow(root, transcription_queue)
    # Create the VicunaChatWindow for the chat interface
    vicuna_chat_window = VicunaChatWindow(root)
    # Start the Tkinter main event loop
    root.mainloop()
