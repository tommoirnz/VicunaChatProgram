# -------------------------------
# Set environment variable so that Tcl/Tk can locate tkdnd files
import os

os.environ["TKDND_LIBRARY"] = r"C:\tkdnd"  # Ensure your tkdnd files are located here
#Best one to day and handles lights with delays etc ask it to turn on light 1 for 10 seconds then turn it off
#Can also run scripts to sequence lights
# -------------------------------
# Import standard libraries
import sys  # System-specific parameters and functions
import threading  # For multi-threading
import queue  # Queue module for thread-safe messaging
import tempfile  # Temporary file creation
import asyncio  # Asynchronous I/O support
from concurrent.futures import ThreadPoolExecutor  # For executing functions asynchronously
import io  # For I/O operations
import re  # Regular expressions for parsing commands

# -------------------------------
# Import tkinter and related modules for GUI
import tkinter as tk
from tkinter import ttk, scrolledtext, filedialog, messagebox

# -------------------------------
# Import audio, numerical and plotting libraries
import sounddevice as sd  # Audio input/output
import numpy as np  # Numerical processing
import speech_recognition as sr  # Speech-to-text (via Google Speech API)
import edge_tts  # For Microsoft Edge Text-to-Speech
from pydub import AudioSegment  # Audio segment manipulation
import simpleaudio as sa  # For playing audio buffers
import matplotlib.pyplot as plt  # Plotting/graphical output

# -------------------------------
# Import PIL for image processing
from PIL import Image, ImageTk  # Image processing and Tkinter-compatible images

# -------------------------------
# Import torch and transformers libraries for running the Vicuna model
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # Tokenizer and model for Vicuna

# -------------------------------
# Import language detection and text-to-speech libraries
from langdetect import detect, LangDetectException  # Detects language from text
import pyttsx3  # Python text-to-speech library (SAPI5)

# -------------------------------
# Import OpenCV for camera capture
import cv2  # Image/video capture

# -------------------------------
# Import Philips Hue control library
from phue import Bridge  # To control Philips Hue bulbs

# -------------------------------
# Optional: Import tkinterdnd2 for drag and drop support
try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
except ImportError:
    DND_FILES = None
    print("tkinterdnd2 not installed; drag and drop will be replaced with a button.")

# -------------------------------
# GLOBAL QUEUES AND VARIABLES
transcription_queue = queue.Queue()  # Stores audio transcription text
vicuna_response_queue = queue.Queue()  # Stores responses from Vicuna
volume_queue = queue.Queue()  # Stores volume meter updates

tts_play_obj = None  # Object to control currently playing TTS audio
last_tts_audio = None  # Tuple to store TTS audio data (raw_data, channels, sample_width, frame_rate)

tts_enabled = True  # Flag indicating if TTS (text-to-speech) is globally enabled
default_edge_voice = None  # Default Edge TTS voice (if any)
default_sapi_voice = None  # Default SAPI TTS voice (if any)

sapi_engine = None  # Global SAPI TTS engine object
sapi_thread = None  # Thread for SAPI TTS processing
sapi_lock = threading.Lock()  # Lock to synchronize access to sapi_engine

vicuna_conversation_history = []  # Global conversation history (as a list of (role, text) tuples)

# Global variable for our simulation window (simulated lights)
light_sim_window = None  # Will be assigned when the Light Simulator window is created

# Global variable for the Philips Hue Bridge connection.
hue_bridge = None  # Will be set after connecting to the Hue Bridge

# -------------------------------
# LANGUAGE OPTIONS for Speech Recognition
LANGUAGE_OPTIONS = [
    "English (US) - en-US", "English (UK) - en-GB", "English (AU) - en-AU",
    "French (France) - fr-FR", "German (Germany) - de-DE", "Italian (Italy) - it-IT",
    "Spanish (Spain) - es-ES", "Chinese (Mandarin) - zh-CN", "Japanese - ja-JP",
    "Russian - ru-RU", "Croatian - hr-HR", "Norwegian - nb-NO", "Swedish - sv-SE",
    "Danish - da-DK", "Irish - ga-IE", "Welsh - cy-GB", "Afrikaans - af-ZA",
    "Thai - th-TH", "Indonesian - id-ID", "Hebrew - he-IL", "Finnish - fi-FI",
    "Hindi - hi-IN", "Arabic - ar-SA", "Dutch - nl-NL", "Polish - pl-PL",
    "Ukrainian - uk-UA", "Latin - la-LA", "Scots Gaelic - gd-GB", "Greek (Greece) - el-GR",
    "Turkish - tr-TR", "Portuguese (Brazil) - pt-BR", "Czech - cs-CZ", "Hungarian - hu-HU"
]

# -------------------------------
# TTS VOICE OPTIONS for Edge TTS
EDGE_VOICE_OPTIONS = ["null",
                      "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
                      "en-GB-LibbyNeural", "en-AU-NatashaNeural", "fr-FR-DeniseNeural",
                      "fr-FR-HenriNeural", "de-DE-KatjaNeural", "de-DE-ConradNeural",
                      "it-IT-IsabellaNeural", "it-IT-LuisaNeural", "es-ES-ElviraNeural",
                      "es-ES-AlonsoNeural", "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
                      "ja-JP-NanamiNeural", "ja-JP-KeitaNeural", "ru-RU-SvetlanaNeural",
                      "hr-HR-RajaNeural", "nb-NO-FinnNeural", "sv-SE-MattiasNeural",
                      "da-DK-ChristelNeural", "ga-IE-ColmNeural", "cy-GB-AledNeural",
                      "af-ZA-WillemNeural", "th-TH-NiwatNeural", "id-ID-ArdiNeural",
                      "he-IL-AvriNeural", "he-IL-AmiraNeural", "fi-FI-NooraNeural",
                      "hi-IN-SwaraNeural", "ar-EG-ShakirNeural", "nl-NL-ColetteNeural",
                      "pl-PL-MarekNeural", "uk-UA-OstapNeural", "el-GR-AthinaNeural",
                      "tr-TR-AhmetNeural", "pt-BR-FranciscaNeural", "cs-CZ-VlastaNeural",
                      "hu-HU-NoemiNeural"]


def get_sapi_voice_options():
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    return ["null"] + [v.name for v in voices]


SAPI_VOICE_OPTIONS = get_sapi_voice_options()
# -------------------------------
# NEW CLASS: CommandScriptRunner
class CommandScriptRunner:
    def __init__(self):
        self.timers = []  # List to hold Timer objects
        self.commands_text = ""  # Loaded script text

    def load_commands(self, filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            self.commands_text = f.read()
        print(f"[CommandScriptRunner] Loaded commands from {filepath}.")

    def run_commands(self, commands_text):
        # Cancel any previously scheduled commands.
        self.stop()
        # Split the commands by the keyword "command" (case-insensitive)
        tokens = re.split(r'\bcommand[s]?\b', commands_text, flags=re.IGNORECASE)
        print("[CommandScriptRunner] Tokens:", tokens)  # Debug print
        cumulative_delay = 0  # cumulative delay in seconds
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            print("[CommandScriptRunner] Processing token:", token)  # Debug print

            # Process timer command: exact match e.g. "timer 10"
            timer_match = re.fullmatch(r'timer\s+(\d+)', token, re.IGNORECASE)
            if timer_match:
                delay_incr = int(timer_match.group(1))
                cumulative_delay += delay_incr
                print(
                    f"[CommandScriptRunner] Timer command: delay increased by {delay_incr} seconds, total delay: {cumulative_delay} seconds")
                continue

            # Process "all lights" command.
            all_match = re.search(r'all\s+lights\s+(on|off)', token, re.IGNORECASE)
            if all_match:
                desired_state = all_match.group(1).lower()

                def command_func_all(state=desired_state):
                    for light_number in light_sim_window.light_states.keys():
                        if light_sim_window.light_states[light_number].lower() != state:
                            light_sim_window.update_light(light_number, state)
                            if light_number in [1, 2]:
                                update_real_light(light_number, state)
                            print(
                                f"[CommandScriptRunner] 'All lights' command executed: Light {light_number} set to {state}.")

                print(f"[CommandScriptRunner] Scheduling 'all lights' command with {cumulative_delay} seconds delay.")
                threading.Timer(cumulative_delay, command_func_all).start()
                continue

            # Process individual light command.
            light_match = re.search(r'light\s+((?:[1-4]|one|won|two|to|too|three|four))\s+(on|off)', token,
                                    re.IGNORECASE)
            if light_match:
                light_str, state = light_match.groups()
                ls_lower = light_str.lower()
                if ls_lower in ['to', 'too', 'two']:
                    light_number = 2
                elif ls_lower in ['one', 'won']:
                    light_number = 1
                elif ls_lower in ['three']:
                    light_number = 3
                elif ls_lower in ['four']:
                    light_number = 4
                else:
                    try:
                        light_number = int(light_str)
                    except Exception:
                        continue

                def command_func(light_num=light_number, state=state):
                    current_state = light_sim_window.light_states.get(light_num, "off")
                    if current_state.lower() != state.lower():
                        light_sim_window.update_light(light_num, state)
                        if light_num in [1, 2]:
                            update_real_light(light_num, state)
                        print(f"[CommandScriptRunner] Command executed: Light {light_num} turned {state.lower()}.")

                print(
                    f"[CommandScriptRunner] Scheduling command for Light {light_number} to turn {state} with {cumulative_delay} seconds delay.")
                threading.Timer(cumulative_delay, command_func).start()
                continue

            # Process natural language command.
            natural_match = re.search(r'(turn|switch)\s+(on|off)\s+light\s+((?:[1-4]|one|won|two|to|too|three|four))',
                                      token, re.IGNORECASE)
            if natural_match:
                _, state, light_str = natural_match.groups()
                ls_lower = light_str.lower()
                if ls_lower in ['to', 'too', 'two']:
                    light_number = 2
                elif ls_lower in ['one', 'won']:
                    light_number = 1
                elif ls_lower in ['three']:
                    light_number = 3
                elif ls_lower in ['four']:
                    light_number = 4
                else:
                    try:
                        light_number = int(light_str)
                    except Exception:
                        continue

                def command_func(light_num=light_number, state=state):
                    current_state = light_sim_window.light_states.get(light_num, "off")
                    if current_state.lower() != state.lower():
                        light_sim_window.update_light(light_num, state)
                        if light_num in [1, 2]:
                            update_real_light(light_num, state)
                        print(
                            f"[CommandScriptRunner] Natural command executed: Light {light_num} turned {state.lower()}.")

                print(
                    f"[CommandScriptRunner] Scheduling natural command for Light {light_number} to turn {state} with {cumulative_delay} seconds delay.")
                threading.Timer(cumulative_delay, command_func).start()
                continue

            print("[CommandScriptRunner] Unrecognized command token:", token)

    def stop(self):
        for t in self.timers:
            t.cancel()
        self.timers = []
        print("[CommandScriptRunner] All scheduled commands have been cancelled.")

    def _parse_light_number(self, token):
        token = token.lower().strip()
        mapping = {
            "one": 1, "won": 1,
            "two": 2, "to": 2, "too": 2,
            "three": 3,
            "four": 4
        }
        if token in mapping:
            return mapping[token]
        try:
            num = int(token)
            if num in [1, 2, 3, 4]:
                return num
        except ValueError:
            return None
        return None


# -------------------------------
# Helper: Render text as LaTeX image using matplotlib.
def render_text_as_latex(text):
    fig = plt.figure(figsize=(8, 6))
    fig.text(0.5, 0.5, text, ha='center', va='center', wrap=True, fontsize=12)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img


# -------------------------------
# Helper: Return matching voice for a language code.
def get_default_voice_for_language(lang_code, voice_options):
    special_mapping = {
        "no": "nb", "he": "he-il", "da": "da-dk", "sv": "sv-se", "la": "it-", "gd": "ga-"
    }
    lang_code = lang_code.lower()
    mapped_code = special_mapping.get(lang_code, lang_code)
    for voice in voice_options:
        if voice.lower().startswith(mapped_code):
            return voice
    return None

#--------------------------------
def separate_commands(cmd_str):
    # Insert a space before any "COMMAND" that is immediately preceded by a non-space character.
    return re.sub(r'(\S)(COMMAND)', r'\1 \2', cmd_str)

# -------------------------------
# NEW HELPER: Preprocess AI responses.
def preprocess_response(response_text):
    cleaned = re.sub(r'(?i)ai:\s*', '', response_text.strip())
    cleaned = re.sub(r'^\s*>+\s*', '', cleaned, flags=re.MULTILINE)
    return cleaned


# -------------------------------
# NEW HELPER: Answer light status.
def answer_light_status():
    states = light_sim_window.light_states  # Get the current state of each simulated light.
    on_lights = [str(i) for i, state in states.items() if state.lower() == "on"]
    if on_lights:
        if len(on_lights) == 1:
            return f"Light {on_lights[0]} is on."
        else:
            return "Lights " + " and ".join(on_lights) + " are on."
    else:
        return "No lights are currently on."


# -------------------------------
# NEW HELPER: Update real Philips Hue light.
def update_real_light(light_number, state):
    # Only update available Hue lights (assume only lights 1 and 2 are available)
    available_hue_lights = [1, 2]
    if hue_bridge is not None and light_number in available_hue_lights:
        if state.lower() == "on":
            hue_bridge.set_light(light_number, 'on', True)
            hue_bridge.set_light(light_number, 'bri', 254)
            print(f"Real light {light_number} turned on.")
        else:
            hue_bridge.set_light(light_number, 'on', False)
            print(f"Real light {light_number} turned off.")


# -------------------------------
# NEW HELPER: Manual switches for direct control.
class ManualSwitches(tk.Frame):
    def __init__(self, master, sim_window):
        super().__init__(master)
        self.sim_window = sim_window
        self.create_switches()

    def create_switches(self):
        for i in range(1, 5):
            btn = tk.Button(self, text=f"Toggle Light {i}", command=lambda i=i: self.toggle_light(i))
            btn.pack(side="left", padx=5, expand=True)

    def toggle_light(self, light_number):
        current_state = self.sim_window.light_states.get(light_number, "off")
        new_state = "off" if current_state.lower() == "on" else "on"
        self.sim_window.update_light(light_number, new_state)
        if light_number in [1, 2]:
            update_real_light(light_number, new_state)


# -------------------------------
# NEW HELPER: Process direct command input.
def process_command_directly(user_text):
    pattern_all = re.compile(r'\bcommands?\s+all\s+lights\s+(on|off)\b', re.IGNORECASE)
    match_all = pattern_all.search(user_text)
    if match_all:
        desired_state = match_all.group(1).lower()
        changed = False
        for light_number, current in light_sim_window.light_states.items():
            if current.lower() != desired_state:
                light_sim_window.update_light(light_number, desired_state)
                if light_number in [1, 2]:
                    update_real_light(light_number, desired_state)
                changed = True
        if changed:
            return f"COMMAND all lights {desired_state}"
        else:
            return f"All lights are already {desired_state}."
    pattern = re.compile(r'\bcommands?\s+light\s+((?:[1-4]|one|won|two|to|too|three|four))\s+(on|off)\b', re.IGNORECASE)
    match = pattern.search(user_text)
    if match:
        light_str, desired_state = match.groups()
        ls_lower = light_str.lower()
        if ls_lower in ['to', 'too', 'two']:
            light_number = 2
        elif ls_lower in ['one', 'won']:
            light_number = 1
        elif ls_lower in ['three']:
            light_number = 3
        elif ls_lower in ['four']:
            light_number = 4
        else:
            try:
                light_number = int(light_str)
            except Exception:
                return None
        current_state = light_sim_window.light_states.get(light_number, "off")
        if current_state.lower() == desired_state.lower():
            return f"Light {light_number} is already {desired_state.lower()}."
        else:
            light_sim_window.update_light(light_number, desired_state)
            if light_number in [1, 2]:
                update_real_light(light_number, desired_state)
            return f"COMMAND light {light_number} {desired_state.lower()}"
    return None


# -------------------------------
# NEW HELPER: Process multiple commands with timer support.
def parse_and_handle_command(response_text):
    response_text = preprocess_response(response_text)
    # Ensure commands are separated by a space.
    response_text = separate_commands(response_text)
    print("Full response for command parsing:", response_text)  # Debug print
    # Split text by occurrences of "command" (case-insensitive)
    tokens = re.split(r'\bcommand[s]?\b', response_text, flags=re.IGNORECASE)
    print("Tokens:", tokens)  # Debug print
    cumulative_delay = 0  # cumulative delay in seconds
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        print("Processing token:", token)  # Debug print

        # Process "all lights" command.
        all_match = re.search(r'all\s+lights\s+(on|off)', token, re.IGNORECASE)
        if all_match:
            desired_state = all_match.group(1).lower()

            def command_func_all(state=desired_state):
                for light_number in light_sim_window.light_states.keys():
                    if light_sim_window.light_states[light_number].lower() != state:
                        light_sim_window.update_light(light_number, state)
                        if light_number in [1, 2]:
                            update_real_light(light_number, state)
                        print(f"'All lights' command executed: Light {light_number} set to {state}.")

            print(f"Scheduling 'all lights' command with {cumulative_delay} seconds delay.")
            threading.Timer(cumulative_delay, command_func_all).start()
            continue

        # Process individual light command.
        light_match = re.search(r'light\s+((?:[1-4]|one|won|two|to|too|three|four))\s+(on|off)', token, re.IGNORECASE)
        if light_match:
            light_str, state = light_match.groups()
            ls_lower = light_str.lower()
            if ls_lower in ['to', 'too', 'two']:
                light_number = 2
            elif ls_lower in ['one', 'won']:
                light_number = 1
            elif ls_lower in ['three']:
                light_number = 3
            elif ls_lower in ['four']:
                light_number = 4
            else:
                try:
                    light_number = int(light_str)
                except Exception:
                    continue

            def command_func(light_num=light_number, state=state):
                current_state = light_sim_window.light_states.get(light_num, "off")
                if current_state.lower() != state.lower():
                    light_sim_window.update_light(light_num, state)
                    if light_num in [1, 2]:
                        update_real_light(light_num, state)
                    print(f"Command executed: Light {light_num} turned {state.lower()}.")

            print(f"Scheduling command for Light {light_number} to turn {state} with {cumulative_delay} seconds delay.")
            threading.Timer(cumulative_delay, command_func).start()
            continue

        # Process natural language command.
        natural_match = re.search(r'(turn|switch)\s+(on|off)\s+light\s+((?:[1-4]|one|won|two|to|too|three|four))',
                                  token, re.IGNORECASE)
        if natural_match:
            _, state, light_str = natural_match.groups()
            ls_lower = light_str.lower()
            if ls_lower in ['to', 'too', 'two']:
                light_number = 2
            elif ls_lower in ['one', 'won']:
                light_number = 1
            elif ls_lower in ['three']:
                light_number = 3
            elif ls_lower in ['four']:
                light_number = 4
            else:
                try:
                    light_number = int(light_str)
                except Exception:
                    continue

            def command_func(light_num=light_number, state=state):
                current_state = light_sim_window.light_states.get(light_num, "off")
                if current_state.lower() != state.lower():
                    light_sim_window.update_light(light_num, state)
                    if light_num in [1, 2]:
                        update_real_light(light_num, state)
                    print(f"Natural command executed: Light {light_num} turned {state.lower()}.")

            print(
                f"Scheduling natural command for Light {light_number} to turn {state} with {cumulative_delay} seconds delay.")
            threading.Timer(cumulative_delay, command_func).start()
            continue

        # Process timer command: now check if the token exactly matches "timer <number>".
        timer_match = re.fullmatch(r'timer\s+(\d+)', token, re.IGNORECASE)
        if timer_match:
            delay_incr = int(timer_match.group(1))
            cumulative_delay += delay_incr
            print(f"Timer command: delay increased by {delay_incr} seconds, total delay: {cumulative_delay} seconds")
            continue

        print("Unrecognized command token:", token)


# -------------------------------
# Microsoft TTS Functions using edge-tts and simpleaudio.
async def _speak_text_async(text, voice, output_file):
    communicator = edge_tts.Communicate(text, voice=voice)
    await communicator.save(output_file)


def speak_text_global(text, output_device):
    global tts_play_obj, last_tts_audio, sapi_thread, sapi_engine
    edge_voice = vicuna_chat_window.edge_voice_combo.get()
    sapi_voice = vicuna_chat_window.sapi_voice_combo.get()
    chosen_system = vicuna_chat_window.tts_system_var.get()
    if not tts_enabled:
        print("TTS is globally disabled.")
        return
    if chosen_system == "edge":
        if edge_voice == "null":
            print("Edge TTS is disabled (voice set to null).")
            return

        def try_tts(chosen_voice):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                temp_filename = f.name
            try:
                asyncio.run(_speak_text_async(text, chosen_voice, temp_filename))
            except Exception as e:
                print(f"Edge TTS error with {chosen_voice}: {e}")
                os.remove(temp_filename)
                return None
            if os.path.getsize(temp_filename) == 0:
                print(f"Edge TTS error: empty file for {chosen_voice}.")
                os.remove(temp_filename)
                return None
            try:
                audio = AudioSegment.from_file(temp_filename, format="mp3")
            except Exception as e:
                print(f"Error loading audio for {chosen_voice}: {e}")
                os.remove(temp_filename)
                return None
            os.remove(temp_filename)
            return audio

        audio = try_tts(edge_voice)
        if audio is None:
            print("Edge TTS failed; no fallback.")
            return
        try:
            last_tts_audio = (audio.raw_data, audio.channels, audio.sample_width, audio.frame_rate)
            if output_device is not None:
                samples = np.array(audio.get_array_of_samples())
                if audio.channels > 1:
                    samples = samples.reshape((-1, audio.channels))
                sd.play(samples, samplerate=audio.frame_rate, device=output_device)
                tts_play_obj = None
            else:
                tts_play_obj = sa.play_buffer(
                    audio.raw_data,
                    num_channels=audio.channels,
                    bytes_per_sample=audio.sample_width,
                    sample_rate=audio.frame_rate
                )
            print("TTS playback started.")
        except Exception as e:
            print("Error playing Edge TTS audio:", e)
    elif chosen_system == "sapi":
        if sapi_voice == "null":
            print("SAPI TTS is disabled (voice set to null).")
            return
        with sapi_lock:
            if sapi_thread and sapi_thread.is_alive():
                stop_sapi()
                sapi_thread.join()

            def speak_sapi_thread(text_to_speak, sapi_voice):
                global sapi_engine
                sapi_engine = pyttsx3.init()
                voices = sapi_engine.getProperty('voices')
                for v in voices:
                    if sapi_voice.lower() in v.name.lower():
                        sapi_engine.setProperty('voice', v.id)
                        break
                sapi_engine.say(text_to_speak)
                try:
                    sapi_engine.runAndWait()
                except RuntimeError as e:
                    print("SAPI RuntimeError:", e)
                try:
                    sapi_engine.stop()
                except Exception:
                    pass
                sapi_engine = None

            sapi_thread = threading.Thread(
                target=speak_sapi_thread,
                args=(text, sapi_voice),
                daemon=True
            )
            sapi_thread.start()
    else:
        print("Unknown TTS system selected.")


def stop_sapi():
    global sapi_engine
    if sapi_engine is not None:
        try:
            sapi_engine.stop()
        except Exception:
            pass
        sapi_engine = None
        print("SAPI TTS stopped.")
    else:
        print("SAPI TTS engine already None.")


def stop_tts():
    global tts_play_obj
    try:
        if tts_play_obj is not None and tts_play_obj.is_playing():
            tts_play_obj.stop()
    except Exception:
        pass
    sd.stop()
    tts_play_obj = None
    stop_sapi()


def replay_tts():
    global last_tts_audio, tts_play_obj
    output_device = None
    try:
        selected_output = vicuna_chat_window.output_combo.get()
        output_device = int(selected_output.split(":")[0])
    except Exception:
        output_device = None
    if last_tts_audio is None:
        print("No TTS audio available to replay.")
        return
    try:
        tts_play_obj = sa.play_buffer(
            last_tts_audio[0],
            num_channels=last_tts_audio[1],
            bytes_per_sample=last_tts_audio[2],
            sample_rate=last_tts_audio[3]
        )
    except Exception as e:
        print("Error replaying TTS audio:", e)


# -------------------------------
# Global conversation history for Vicuna Chat
vicuna_conversation_history = []

from datetime import datetime


# -------------------------------
# Updated vicuna_chat_response: Build prompt including current light states,
# pass it to the Vicuna model and return the generated response.
def vicuna_chat_response(user_text, temperature=0.9, top_p=0.9, rep_penalty=1.1):
    global vicuna_conversation_history, beam_search_spinbox
    current_dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    light_status = ""
    if light_sim_window:
        light_status = ", ".join([f"Light {i} is {state}" for i, state in light_sim_window.light_states.items()])
    system_prompt = (
        f"system: You are an advanced AI assistant with extensive knowledge in many areas—including science, history, and mathematics. "
        f"Today's date and time is {current_dt}. "
        "You can control a simulated lighting system, and you are also capable of analyzing image descriptions. "
        "When you receive a command regarding the lights, output a command sequence starting with 'COMMAND'. "
        "Valid commands include 'COMMAND light X on', 'COMMAND light X off', and 'COMMAND all lights on' or 'COMMAND all lights off'. "
        "If a delay is needed between commands, include timer commands using the format 'COMMAND timer X', where X is the number of seconds to wait before executing the next command. "
        "Important: Do not insert an initial timer command unless explicitly requested. "
        "Additionally, when you receive a message that starts with 'BLIP:', interpret it as an image caption generated by the BLIP module. "
        "Provide a thoughtful commentary or ask clarifying questions about the image. For example, if you receive 'BLIP: a dog playing in the park on a sunny day', "
        "you might reply 'That sounds like a lively scene! Can you tell me more about the environment or the dog?' "
        "For all other queries, provide a full natural language answer."
    )

    if light_status:
        system_prompt += f" Current light states: {light_status}."
    if vicuna_conversation_history and vicuna_conversation_history[0][0] == "system":
        vicuna_conversation_history[0] = ("system", system_prompt)
    else:
        vicuna_conversation_history.insert(0, ("system", system_prompt))
    vicuna_conversation_history.append(("user", user_text))
    prompt = "\n".join(f"{role}: {text}" for role, text in vicuna_conversation_history) + "\nAI:"
    input_ids = vicuna_tokenizer(prompt, return_tensors="pt").input_ids.to(vicuna_model.device)
    num_beams = int(beam_search_spinbox.get())
    with torch.no_grad():
        max_total_tokens = 16384
        output = vicuna_model.generate(
            input_ids,
            max_length=max_total_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=rep_penalty,
            num_beams=num_beams,
            do_sample=True
        )
    full_response = vicuna_tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if "AI:" in full_response:
        latest_response = full_response.split("AI:")[-1].strip()
    else:
        latest_response = full_response.strip()
    vicuna_conversation_history.append(("ai", latest_response))
    return latest_response


# -------------------------------
# -------------------------------
# New Light Simulator Window with Script Controls
class LightSimulatorWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Light Simulator")
        self.geometry("600x300")
        self.canvas = tk.Canvas(self, bg="white", height=150)
        self.canvas.pack(expand=True, fill="both")
        self.light_ids = {}
        # Initialize all 4 lights to "off"
        self.light_states = {1: "off", 2: "off", 3: "off", 4: "off"}
        self.light_radius = 40
        spacing = 20
        for i in range(1, 5):
            x = spacing + (i - 0.5) * (2 * self.light_radius + spacing)
            y = 75
            light_id = self.canvas.create_oval(
                x - self.light_radius, y - self.light_radius,
                x + self.light_radius, y + self.light_radius,
                fill="black", outline="gray", width=2
            )
            self.canvas.create_text(x, y + self.light_radius + 15, text=f"Light {i}")
            self.light_ids[i] = light_id

        # Create a control frame at bottom for manual switches and script control
        control_frame = tk.Frame(self)
        control_frame.pack(side="bottom", fill="x", pady=10)

        # Manual control buttons
        manual_frame = tk.Frame(control_frame)
        manual_frame.pack(side="top", fill="x", pady=5)
        self.manual_switches = ManualSwitches(manual_frame, self)
        self.manual_switches.pack(side="left")

        # Script control buttons
        script_frame = tk.Frame(control_frame)
        script_frame.pack(side="bottom", fill="x", pady=5)
        self.load_script_button = tk.Button(script_frame, text="Load Script", command=self.load_script)
        self.load_script_button.pack(side="left", padx=5)
        self.run_script_button = tk.Button(script_frame, text="Run Script", command=self.run_script)
        self.run_script_button.pack(side="left", padx=5)
        self.stop_script_button = tk.Button(script_frame, text="Stop Script", command=self.stop_script)
        self.stop_script_button.pack(side="left", padx=5)

        # Instantiate our command script runner
        self.script_runner = CommandScriptRunner()

    def toggle_light(self, light_number):
        current = self.light_states.get(light_number, "off")
        new_state = "off" if current.lower() == "on" else "on"
        self.update_light(light_number, new_state)
        update_real_light(light_number, new_state)

    def update_light(self, light_number, state):
        if light_number in self.light_ids:
            fill_color = "yellow" if state.lower() == "on" else "black"
            self.canvas.itemconfig(self.light_ids[light_number], fill=fill_color)
            self.light_states[light_number] = state.lower()
            print(f"Updated Light {light_number}: {state.lower()}")
        else:
            print("Light number out of range:", light_number)

    def load_script(self):
        filepath = filedialog.askopenfilename(
            title="Select Command Script File",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        if filepath:
            try:
                self.script_runner.load_commands(filepath)
                messagebox.showinfo("Script Loaded", "Command script loaded successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load script: {e}")

    def run_script(self):
        if not self.script_runner.commands_text:
            messagebox.showwarning("No Script", "No command script loaded!")
            return
        threading.Thread(target=self.script_runner.run_commands, args=(self.script_runner.commands_text,), daemon=True).start()

    def stop_script(self):
        self.script_runner.stop()
        messagebox.showinfo("Script Stopped", "Scheduled commands have been cancelled.")

# -------------------------------
# Combined ASR Application (Main Tkinter Window)
class CombinedASRApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Speech Recognition - English Only")
        self.root.configure(bg="white")
        self.root.geometry("400x700")
        self.root.bind_all("<Control-Button-1>", lambda event: self.toggle_record_and_submit())
        self.samplerate = 16000
        self.chunk_size = 2048
        self.gain = 1.0
        self.buffered_chunks = []
        self.spoken_language_code = "en-US"
        self.buffer_size_var = tk.IntVar(value=100)
        self.overlap_percentage = 4
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.is_recording = False
        self.build_widgets()
        self.root.after(100, self.process_volume_queue)

    def build_widgets(self):
        tk.Label(self.root, text="Select Microphone Device:", bg="white").pack(pady=5)
        self.device_combobox = ttk.Combobox(self.root, state="readonly", width=50)
        self.device_combobox.pack(pady=5)
        self.list_audio_devices()
        tk.Label(self.root, text="Select Input Language:", bg="white").pack(pady=5)
        self.language_combobox = ttk.Combobox(self.root, state="readonly", width=50, values=LANGUAGE_OPTIONS)
        self.language_combobox.pack(pady=5)
        self.language_combobox.current(0)
        self.language_combobox.bind("<<ComboboxSelected>>", self.update_language)
        tk.Label(self.root, text="Mic Gain:", bg="white").pack(pady=5)
        self.gain_slider = tk.Scale(self.root, from_=1.0, to=4.0, resolution=0.1,
                                    orient="horizontal", length=225, command=self.set_gain)
        self.gain_slider.set(1.0)
        self.gain_slider.pack(pady=5)
        frame_buf = tk.Frame(self.root, bg="white")
        frame_buf.pack(pady=5)
        tk.Label(frame_buf, text="Buffer Size (chunks):", bg="white").pack(side="left", padx=5)
        self.buffer_size_slider = tk.Scale(frame_buf, from_=20, to=140, resolution=10,
                                           orient="horizontal", variable=self.buffer_size_var, length=200)
        self.buffer_size_slider.pack(side="left", padx=5)
        frame_overlap = tk.Frame(self.root, bg="white")
        frame_overlap.pack(pady=5)
        tk.Label(frame_overlap, text="Overlap (%):", bg="white").pack(side="left", padx=5)
        self.overlap_slider = tk.Scale(frame_overlap, from_=0, to=20, resolution=1,
                                       orient="horizontal", length=200, command=self.set_overlap)
        self.overlap_slider.set(self.overlap_percentage)
        self.overlap_slider.pack(side="left", padx=5)
        tk.Label(self.root, text="Volume:", bg="white").pack(pady=5)
        self.volume_bar = ttk.Progressbar(self.root, orient="horizontal",
                                          mode="determinate", maximum=100, length=375)
        self.volume_bar.pack(pady=5)
        tk.Label(self.root, text="Beam Search (1-9):", bg="white").pack(pady=5)
        global beam_search_spinbox
        beam_search_spinbox = tk.Spinbox(self.root, from_=1, to=9, width=5)
        beam_search_spinbox.pack(pady=5)
        self.toggle_button = tk.Button(self.root, text="Record & Submit",
                                       command=self.toggle_record_and_submit,
                                       font=("Helvetica", 14), bg="#4CAF50", fg="white", bd=3, relief="raised")
        self.toggle_button.pack(pady=10, fill="x", ipady=10)
        self.tts_toggle_button = tk.Button(self.root, text="Disable TTS", command=self.toggle_tts,
                                           font=("Helvetica", 12), bg="#9C27B0", fg="white", bd=3, relief="raised")
        self.tts_toggle_button.pack(pady=5, fill="x", ipady=5)
        self.clear_text_button = tk.Button(self.root, text="Clear Transcript Text",
                                           command=self.clear_transcript_text,
                                           font=("Helvetica", 12), bg="#F44336", fg="white", bd=3, relief="raised")
        self.clear_text_button.pack(pady=5, fill="x", ipady=5)

    def toggle_tts(self):
        global tts_enabled, default_edge_voice, default_sapi_voice
        tts_enabled = not tts_enabled
        if tts_enabled:
            self.tts_toggle_button.config(text="Disable TTS")
            if default_edge_voice:
                vicuna_chat_window.edge_voice_combo.set(default_edge_voice)
            if default_sapi_voice:
                vicuna_chat_window.sapi_voice_combo.set(default_sapi_voice)
        else:
            self.tts_toggle_button.config(text="Enable TTS")
            default_edge_voice = vicuna_chat_window.edge_voice_combo.get()
            default_sapi_voice = vicuna_chat_window.sapi_voice_combo.get()
            vicuna_chat_window.edge_voice_combo.set("null")
            vicuna_chat_window.sapi_voice_combo.set("null")

    def clear_transcript_text(self):
        if transcript_window:
            transcript_window.text_widget.delete("1.0", tk.END)

    def update_language(self, event=None):
        selected = self.language_combobox.get()
        self.spoken_language_code = selected.split("-")[-1].strip()

    def list_audio_devices(self):
        devices = sd.query_devices()
        device_list = []
        for i, dev in enumerate(devices):
            if dev['max_input_channels'] > 0:
                device_list.append(f"{i}: {dev['name']}")
        self.device_combobox['values'] = device_list
        if device_list:
            self.device_combobox.current(0)

    def set_gain(self, value):
        try:
            self.gain = float(value)
        except Exception as e:
            print("Error setting gain:", e)

    def set_overlap(self, value):
        try:
            self.overlap_percentage = float(value)
        except Exception as e:
            print("Error setting overlap:", e)

    def process_volume_queue(self):
        try:
            while not volume_queue.empty():
                vol = volume_queue.get_nowait()
                self.volume_bar['value'] = vol
        except queue.Empty:
            pass
        self.root.after(100, self.process_volume_queue)

    def audio_callback(self, indata, frames, time_info, status):
        try:
            if status:
                print("Audio status:", status)
            indata = indata * self.gain
            silence_threshold = 0.02
            if np.linalg.norm(indata) < silence_threshold:
                return
            self.buffered_chunks.append(indata.copy())
            vol = np.linalg.norm(indata) * 10
            volume_queue.put(min(vol, 100))
            current_buf_size = self.buffer_size_var.get()
            if len(self.buffered_chunks) >= current_buf_size:
                self.executor.submit(self.worker_thread, self.buffered_chunks.copy())
                overlap = self.overlap_percentage / 100.0
                retain = int(overlap * len(self.buffered_chunks))
                self.buffered_chunks = self.buffered_chunks[-retain:] if retain > 0 else []
        except Exception as e:
            print("Error in audio callback:", e)

    def worker_thread(self, audio_chunks):
        self.process_audio_buffer(audio_chunks)

    def process_audio_buffer(self, audio_chunks):
        try:
            combined_audio = np.concatenate(audio_chunks, axis=0).astype(np.float32)
            audio_int16 = np.int16(combined_audio * 32767)
            audio_bytes = audio_int16.tobytes()
            audio_data = sr.AudioData(audio_bytes, self.samplerate, 2)
            detected_language, text = detect_language_from_audio(audio_data, language_code=self.spoken_language_code)
            if detected_language:
                print(f"Detected Language: {detected_language}")
                self.spoken_language_code = detected_language
            if text.strip():
                transcription_queue.put(text + "\n")
            return text
        except Exception as e:
            transcription_queue.put("Error processing audio: " + str(e) + "\n")
            return ""

    def toggle_record_and_submit(self):
        if not self.is_recording:
            self.buffered_chunks = []
            try:
                mic_index = int(self.device_combobox.get().split(":")[0])
            except Exception as e:
                messagebox.showerror("Device Error", "No microphone device selected.")
                return
            try:
                self.stream = sd.InputStream(samplerate=self.samplerate, channels=1, device=mic_index,
                                             blocksize=self.chunk_size, callback=self.audio_callback)
                self.stream.start()
            except Exception as e:
                messagebox.showerror("Audio Stream Error", str(e))
                return
            self.is_recording = True
            self.toggle_button.config(text="Stop & Submit")
            self.root.configure(bg="red")
        else:
            if hasattr(self, "stream"):
                self.stream.stop()
            self.is_recording = False
            self.toggle_button.config(text="Record & Submit")
            self.root.configure(bg="white")
            transcript = self.process_audio_buffer(self.buffered_chunks)
            self.buffered_chunks = []
            vicuna_chat_window.insert_user_text(transcript)


def detect_language_from_audio(audio_data, language_code="en-US"):
    recognizer = sr.Recognizer()
    try:
        text = recognizer.recognize_google(audio_data, language=language_code)
        detected_language = detect(text)
        return detected_language, text
    except sr.UnknownValueError:
        print("Speech Recognition could not understand the audio.")
        return None, ""
    except sr.RequestError as e:
        print(f"Could not request results from Speech Recognition service; {e}")
        return None, ""
    except LangDetectException as e:
        print(f"Language detection failed: {e}")
        return None, ""


# -------------------------------
# Transcript Window
class TranscriptWindow(tk.Toplevel):
    def __init__(self, master, queue):
        super().__init__(master)
        self.title("Transcript")
        self.geometry("500x400")
        self.text_widget = scrolledtext.ScrolledText(self, wrap="word", font=("Arial", 16))
        self.text_widget.pack(expand=True, fill="both")
        self.font_size_slider = ttk.Scale(self, from_=8, to=48, orient="horizontal", command=self.change_font_size)
        self.font_size_slider.set(16)
        self.font_size_slider.pack(fill="x")
        self.save_button = tk.Button(self, text="Save Transcript", command=self.save_transcript)
        self.save_button.pack(pady=5)
        self.queue = queue
        self.after(100, self.update_text)

    def change_font_size(self, value):
        new_size = int(float(value))
        self.text_widget.config(font=("Arial", new_size))
        self.update_idletasks()

    def update_text(self):
        while not self.queue.empty():
            msg = self.queue.get()
            if msg == "__CLEAR__":
                self.text_widget.delete("1.0", tk.END)
            else:
                self.text_widget.insert(tk.END, msg)
                self.text_widget.see(tk.END)
        self.after(100, self.update_text)

    def save_transcript(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.text_widget.get("1.0", tk.END))


# -------------------------------
# Vicuna Chat Window
class VicunaChatWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Vicuna Chat")
        top_controls_frame = tk.Frame(self)
        top_controls_frame.pack(side="top", fill="x", padx=10, pady=5)
        self.clear_chat_button = tk.Button(top_controls_frame, text="Clear Chat", command=self.clear_chat)
        self.clear_chat_button.pack(side="left", padx=5)
        self.clear_history_button = tk.Button(top_controls_frame, text="Clear History", command=self.clear_history)
        self.clear_history_button.pack(side="left", padx=5)
        options_frame = tk.Frame(self)
        options_frame.pack(side="top", fill="x", padx=10, pady=5)
        tk.Label(options_frame, text="Edge TTS Voice:").grid(row=0, column=0, padx=5)
        self.edge_voice_combo = ttk.Combobox(options_frame, values=EDGE_VOICE_OPTIONS, state="readonly", width=20)
        self.edge_voice_combo.grid(row=0, column=1, padx=5)
        self.edge_voice_combo.current(1)
        tk.Label(options_frame, text="SAPI5 Voice:").grid(row=0, column=2, padx=5)
        self.sapi_voice_combo = ttk.Combobox(options_frame, values=SAPI_VOICE_OPTIONS, state="readonly", width=20)
        self.sapi_voice_combo.grid(row=0, column=3, padx=5)
        self.sapi_voice_combo.current(0)
        self.edge_voice_combo.bind("<<ComboboxSelected>>", self.on_edge_voice_change)
        self.sapi_voice_combo.bind("<<ComboboxSelected>>", self.on_sapi_voice_change)
        tk.Label(options_frame, text="TTS System:").grid(row=0, column=4, padx=5)
        self.tts_system_var = tk.StringVar(value="edge")
        tk.Radiobutton(options_frame, text="Edge TTS", variable=self.tts_system_var, value="edge").grid(row=0, column=5,
                                                                                                        padx=5)
        tk.Radiobutton(options_frame, text="SAPI TTS", variable=self.tts_system_var, value="sapi").grid(row=0, column=6,
                                                                                                        padx=5)
        tk.Label(options_frame, text="Output Device:").grid(row=0, column=7, padx=5)
        output_devices = [f"{i}: {d['name']}" for i, d in enumerate(sd.query_devices()) if d['max_output_channels'] > 0]
        self.output_combo = ttk.Combobox(options_frame, values=output_devices, state="readonly", width=20)
        self.output_combo.grid(row=0, column=8, padx=5)
        if output_devices:
            self.output_combo.current(0)
        self.auto_tts_var = tk.BooleanVar(value=True)
        self.auto_tts_check = tk.Checkbutton(options_frame, text="Auto TTS", variable=self.auto_tts_var)
        self.auto_tts_check.grid(row=0, column=9, padx=5)
        self.stop_tts_button = tk.Button(options_frame, text="Stop TTS", command=stop_tts)
        self.stop_tts_button.grid(row=0, column=10, padx=5)
        self.replay_tts_button = tk.Button(options_frame, text="Replay TTS", command=replay_tts)
        self.replay_tts_button.grid(row=0, column=11, padx=5)
        self.chat_display = scrolledtext.ScrolledText(self, wrap="word", font=("Arial", 12),
                                                      bg="black", fg="white", insertbackground="white")
        self.chat_display.pack(expand=True, fill="both")
        self.font_size_slider = ttk.Scale(self, from_=8, to=48, orient="horizontal", command=self.change_text_size)
        self.font_size_slider.set(12)
        self.font_size_slider.pack(fill="x")
        self.save_chat_button = tk.Button(self, text="Save Chat Transcript", command=self.save_chat_transcript)
        self.save_chat_button.pack(pady=5)
        self.render_chat_button = tk.Button(self, text="Render Chat as LaTeX", command=self.render_chat_as_latex,
                                            font=("Helvetica", 12), bg="#FF9800", fg="white", bd=3, relief="raised")
        self.render_chat_button.pack(pady=5, fill="x")

        # Input frame with text box, submit and paste buttons.
        input_frame = tk.Frame(self)
        input_frame.pack(side="bottom", fill="x", padx=10, pady=5)
        self.user_input = tk.Text(input_frame, height=4, font=("Arial", 12))
        self.user_input.pack(fill="x", padx=10, pady=(0, 5))
        btn_frame = tk.Frame(input_frame)
        btn_frame.pack(fill="x")
        self.submit_button = tk.Button(btn_frame, text="Submit", command=self.on_submit)
        self.submit_button.pack(side="left", padx=5, pady=(0, 5))
        self.paste_button = tk.Button(btn_frame, text="Paste", command=self.paste_from_clipboard)
        self.paste_button.pack(side="left", padx=5, pady=(0, 5))
        self.after(100, self.check_vicuna_response_queue)

    def paste_from_clipboard(self):
        try:
            clipboard_text = self.clipboard_get()
            self.user_input.insert(tk.END, clipboard_text)
        except Exception as e:
            messagebox.showerror("Clipboard Error", f"Unable to paste text: {e}")

    def on_edge_voice_change(self, event):
        if self.edge_voice_combo.get() == "null" and self.sapi_voice_combo.get() == "null":
            if len(SAPI_VOICE_OPTIONS) > 1:
                self.sapi_voice_combo.current(1)

    def on_sapi_voice_change(self, event):
        if self.sapi_voice_combo.get() == "null" and self.edge_voice_combo.get() == "null":
            if len(EDGE_VOICE_OPTIONS) > 1:
                self.edge_voice_combo.current(1)

    def clear_chat(self):
        self.chat_display.delete("1.0", tk.END)

    def clear_history(self):
        self.clear_chat()
        global vicuna_conversation_history
        vicuna_conversation_history = []

    def render_chat_as_latex(self):
        chat_text = self.chat_display.get("1.0", tk.END)
        try:
            img = render_text_as_latex(chat_text)
            window = tk.Toplevel(self)
            window.title("Rendered Chat")
            photo = ImageTk.PhotoImage(img)
            label = tk.Label(window, image=photo)
            label.image = photo
            label.pack()
        except Exception as e:
            messagebox.showerror("Rendering Error", f"Error rendering chat as LaTeX: {e}")

    def change_text_size(self, value):
        new_size = int(float(value))
        self.chat_display.config(font=("Arial", new_size))
        self.update_idletasks()

    def save_chat_transcript(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                                 filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.chat_display.get("1.0", tk.END))

    def on_submit(self, event=None):
        text = self.user_input.get("1.0", tk.END).strip()
        if text:
            lower_text = text.lower()
            # Modified: if the command contains 'timer', use the timer-enabled parser
            if lower_text.startswith("command") or lower_text.startswith("commands"):
                if "timer" in lower_text:
                    parse_and_handle_command(text)
                    self.append_text("Processed timer command sequence.\n\n")
                    self.user_input.delete("1.0", tk.END)
                    return
                else:
                    direct_response = process_command_directly(text)
                    if direct_response:
                        self.append_text(direct_response + "\n\n")
                        if self.auto_tts_var.get():
                            try:
                                output_device_index = int(self.output_combo.get().split(":")[0])
                            except Exception as e:
                                print("Error parsing output device:", e)
                                output_device_index = None
                            threading.Thread(target=speak_text_global, args=(direct_response, output_device_index),
                                             daemon=True).start()
                        self.user_input.delete("1.0", tk.END)
                        return
            self.append_text("User: " + text + "\n")
            self.user_input.delete("1.0", tk.END)
            threading.Thread(target=self.process_vicuna_input, args=(text,), daemon=True).start()

    def process_vicuna_input(self, text):
        response = vicuna_chat_response(text)
        vicuna_response_queue.put(response)

    def check_vicuna_response_queue(self):
        try:
            while True:
                response = vicuna_response_queue.get_nowait()
                cleaned_response = preprocess_response(response)
                self.append_text(cleaned_response + "\n\n")
                try:
                    lang = detect(cleaned_response)
                    default_voice = get_default_voice_for_language(lang, self.edge_voice_combo["values"])
                    if default_voice:
                        for idx, voice in enumerate(self.edge_voice_combo["values"]):
                            if voice == default_voice:
                                self.edge_voice_combo.current(idx)
                                break
                except Exception as e:
                    print("Language detection error:", e)
                if self.auto_tts_var.get():
                    try:
                        output_device_index = int(self.output_combo.get().split(":")[0])
                    except Exception as e:
                        print("Error parsing output device:", e)
                        output_device_index = None
                    threading.Thread(target=speak_text_global, args=(cleaned_response, output_device_index),
                                     daemon=True).start()
                parse_and_handle_command(cleaned_response)
        except queue.Empty:
            pass
        self.after(100, self.check_vicuna_response_queue)

    def append_text(self, text):
        self.chat_display.insert(tk.END, text)
        self.chat_display.see(tk.END)

    def insert_user_text(self, text):
        self.user_input.delete("1.0", tk.END)
        self.user_input.insert("1.0", text)
        self.on_submit()


# -------------------------------
# New Tkinter Window: Image Drop for BLIP Captioning with Camera Capture.
blip_processor = None
blip_model = None


class ImageDropWindow(tk.Toplevel):
    def __init__(self, master, vicuna_chat_window):
        super().__init__(master)
        self.title("Image Drop / Camera Capture for BLIP Captioning")
        self.geometry("600x500")
        self.vicuna_chat_window = vicuna_chat_window
        self.drop_frame = tk.Frame(self)
        self.drop_frame.pack(expand=True, fill="both", padx=10, pady=10)
        if DND_FILES is not None:
            self.drop_area = tk.Label(self.drop_frame, text="Drag and drop an image file here", bg="lightgray",
                                      relief="sunken")
            self.drop_area.pack(expand=True, fill="both")
            try:
                self.drop_area.drop_target_register(DND_FILES)
                self.drop_area.dnd_bind('<<Drop>>', self.drop)
            except Exception as e:
                print("Drag-and-drop registration failed:", e)
                self.drop_area.config(text="Click to load image")
                self.drop_area.bind("<Button-1>", lambda event: self.load_image())
        else:
            self.drop_area = tk.Button(self.drop_frame, text="Click to load image", command=self.load_image)
            self.drop_area.pack(expand=True, fill="both")
        cam_frame = tk.Frame(self)
        cam_frame.pack(pady=5)
        tk.Label(cam_frame, text="Select Camera:").pack(side="left", padx=5)
        self.camera_combobox = ttk.Combobox(cam_frame, state="readonly", width=10, values=["0", "1", "2"])
        self.camera_combobox.pack(side="left", padx=5)
        self.camera_combobox.current(0)
        self.capture_button = tk.Button(cam_frame, text="Capture from Camera", command=self.capture_from_camera)
        self.capture_button.pack(side="left", padx=5)
        self.image_label = tk.Label(self)
        self.image_label.pack(pady=10)
        self.caption_text = tk.Text(self, height=3, wrap="word")
        self.caption_text.pack(pady=5, fill="x")
        self.send_button = tk.Button(self, text="Send Caption to Vicuna", command=self.send_caption)
        self.send_button.pack(pady=5)
        self.current_caption = None

    def drop(self, event):
        file_path = event.data
        if file_path.startswith("{") and file_path.endswith("}"):
            file_path = file_path[1:-1]
        self.process_image(file_path)

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png;*.bmp;*.gif")])
        if file_path:
            self.process_image(file_path)

    def capture_from_camera(self):
        cam_index = self.camera_combobox.get()
        cap = cv2.VideoCapture(int(cam_index))
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            self.process_image_from_object(image)
        else:
            messagebox.showerror("Camera Error", f"Could not capture image from camera {cam_index}.")

    def process_image_from_object(self, image):
        global blip_processor, blip_model
        if blip_processor is None or blip_model is None:
            try:
                from transformers import BlipProcessor, BlipForConditionalGeneration
                cache_dir = r"C:\Users\tomsp\.cache\huggingface\hub\models--Salesforce--blip-image-captioning-base"
                blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base",
                                                               cache_dir=cache_dir)
                blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base",
                                                                          cache_dir=cache_dir)
            except Exception as e:
                messagebox.showerror("BLIP Error", f"Failed to load BLIP model:\n{e}")
                return
        try:
            inputs = blip_processor(images=image, return_tensors="pt")
            out = blip_model.generate(**inputs)
            caption = blip_processor.decode(out[0], skip_special_tokens=True)
            combined_caption = caption
        except Exception as e:
            messagebox.showerror("BLIP Error", f"Failed to process image with BLIP:\n{e}")
            return
        self.display_image_and_caption(image, combined_caption)

    def process_image(self, file_path):
        try:
            image = Image.open(file_path).convert("RGB")
        except Exception as e:
            messagebox.showerror("Image Error", f"Failed to open image:\n{e}")
            return
        self.process_image_from_object(image)

    def display_image_and_caption(self, image, combined_caption):
        self.photo = ImageTk.PhotoImage(image.resize((300, 300)))
        self.image_label.config(image=self.photo)
        self.caption_text.delete("1.0", tk.END)
        self.caption_text.insert(tk.END, combined_caption)
        self.current_caption = combined_caption

    def send_caption(self):
        if self.current_caption:
           # self.vicuna_chat_window.insert_user_text(self.current_caption)
           self.vicuna_chat_window.insert_user_text("BLIP: " + self.current_caption)

           messagebox.showinfo("Caption Sent", "Caption sent to Vicuna Chat.")


# -------------------------------
# VICUNA MODEL SETUP
vicuna_model_path = r"C:\models\vicuna-13b-v1.5-16k"
print("Loading Vicuna Tokenizer...")
vicuna_tokenizer = AutoTokenizer.from_pretrained(vicuna_model_path, use_fast=False)
print("Loading Vicuna Model in 4-bit mode...")
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
if __name__ == "__main__":
    BRIDGE_IP = "192.168.178.197"  # Change this to your actual Hue Bridge IP.
    try:
        hue_bridge = Bridge(BRIDGE_IP)
        hue_bridge.connect()  # Press the bridge button when prompted.
        print(f"Connected to Hue Bridge at {BRIDGE_IP}")
    except Exception as e:
        print("Hue Bridge connection error:", e)
        hue_bridge = None

    root = tk.Tk()
    root.tk.eval('lappend auto_path "C:/tkdnd"')
    try:
        print("tkdnd version:", root.tk.eval('package require tkdnd'))
    except Exception as e:
        print("Error loading tkdnd:", e)

    app = CombinedASRApp(root)
    transcript_window = TranscriptWindow(root, transcription_queue)
    vicuna_chat_window = VicunaChatWindow(root)
    image_drop_window = ImageDropWindow(root, vicuna_chat_window)

    # Use the new Light Simulator window
    light_sim_window = LightSimulatorWindow(root)

    # Optionally, turn off physical Hue lights at startup.
    if hue_bridge is not None:
        update_real_light(1, "off")
        update_real_light(2, "off")

    # (Optional) Uncomment the following line to generate the binary counter script automatically.
    # generate_binary_count_script("binary_count.txt")

    root.mainloop()
