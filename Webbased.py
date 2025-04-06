import sys
import asyncio

# -----------------------------------------------------
# Set Windows-specific asyncio event loop policy.
# This is required on Windows to avoid issues with the default event loop.
if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# -----------------------------------------------------
# -----------------------------------------------------
# Additional information: (28/3/2025)
# This version has Stable Diffusion 1.5, Vicuna 13B,
# speech recognition using Whisper, and image captioning via BLIP.

import os

# Disable torch.compile optimizations via an environment variable.
os.environ["USE_TORCH_COMPILE"] = "0"

import torch

# -----------------------------------------------------
# Patch torch.compiler if it doesn't exist
# This ensures that our code using torch.compile can run even if torch.compiler is missing.
if not hasattr(torch, "compiler"):
    torch.compiler = torch.compile

# Patch torch.compiler.disable if missing so that the decorator does nothing.
# This prevents errors when using the disable decorator on environments that do not support it.
if not hasattr(torch.compiler, "disable"):
    def disable(recursive=False):
        def decorator(fn):
            return fn

        return decorator


    torch.compiler.disable = disable

# -----------------------------------------------------
# Patch missing float8 attributes for diffusers.
# These attributes may not exist in some versions of PyTorch, so we substitute them with float16.
if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float16
if not hasattr(torch, "float8_e5m2"):
    torch.float8_e5m2 = torch.float16

# Print out some important torch and CUDA configuration information.
print("torch.version.cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
print("Has torch.compile:", hasattr(torch, "compile"))
print("Has torch.compiler:", hasattr(torch, "compiler"))
print("Has torch.compiler.disable:", hasattr(torch.compiler, "disable"))
print("torch.float8_e4m3fn:", torch.float8_e4m3fn)
print("torch.float8_e5m2:", torch.float8_e5m2)

# -----------------------------------------------------
# Patch huggingface_hub if needed.
# Some older versions might not have the cached_download function so we alias it.
import huggingface_hub

if not hasattr(huggingface_hub, "cached_download"):
    huggingface_hub.cached_download = huggingface_hub.hf_hub_download
print("huggingface_hub version:", huggingface_hub.__version__)

# -----------------------------------------------------
# Load both text-to-image and image-to-image Stable Diffusion pipelines.
# We import the necessary diffusers pipelines and PIL for image handling.
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
import PIL.Image as Image

# Determine device: use "cuda" if available, otherwise fallback to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------------------------------
# Loading the text-to-image pipeline.
# Here, we use the RunwayML version of Stable Diffusion v1.5.
print("Loading text2img pipeline...")
pipe_text2img = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Updated model identifier for text-to-image
    cache_dir=r"C:\Users\tomsp\AiImage1\models",  # Local cache directory
    local_files_only=True,  # Use local files only (no downloading)
    torch_dtype=torch.float16  # Use float16 precision for performance
).to(device)

# -----------------------------------------------------
# Loading the image-to-image pipeline.
# This pipeline transforms an existing image based on a text prompt.
print("Loading img2img pipeline...")
pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",  # Updated model identifier for image-to-image
    cache_dir=r"C:\Users\tomsp\AiImage1\models",
    local_files_only=True,
    torch_dtype=torch.float16
).to(device)


# -----------------------------------------------------
# Define function to generate an image from text (text-to-image).
def generate_text2img(prompt, steps=200, guidance=7.5):
    """
    Generate a new image from scratch using only a text prompt.

    Args:
        prompt (str): The text prompt to generate the image.
        steps (int): Number of inference steps.
        guidance (float): Guidance scale for image generation.

    Returns:
        image (PIL.Image): The generated image.
    """
    with torch.no_grad():
        image = pipe_text2img(
            prompt=prompt,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]
    return image


# -----------------------------------------------------
# Define function to generate an image based on an initial image (img2img).
def generate_img2img(prompt, init_image, strength=0.7, steps=50, guidance=7.5):
    """
    Transform an existing image (init_image) according to the text prompt.
    Resizes the initial image to 512x512 for consistency.

    Args:
        prompt (str): The text prompt for image transformation.
        init_image (PIL.Image): The initial image to transform.
        strength (float): Strength of the transformation.
        steps (int): Number of inference steps.
        guidance (float): Guidance scale for transformation.

    Returns:
        image (PIL.Image): The transformed image.
    """
    # Resize the image for consistent input dimensions.
    init_image = init_image.resize((512, 512))
    with torch.no_grad():
        image = pipe_img2img(
            prompt=prompt,
            image=init_image,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=guidance
        ).images[0]
    return image


# -----------------------------------------------------
# Import re module for prompt refinement.
import re


# Define a function that checks for the "draw" keyword in user text.
def maybe_run_stable_diffusion(user_text, user_image):
    """
    Determines whether to run Stable Diffusion based on the user's text.
    If the keyword 'draw' is present:
      - Removes the keyword "draw" from the prompt.
      - Uses image-to-image if an image is provided, otherwise text-to-image.

    Args:
        user_text (str): User's input text.
        user_image (PIL.Image or None): Optional image provided by the user.

    Returns:
        PIL.Image or None: Generated image from Stable Diffusion, or None if not applicable.
    """
    if "draw" in user_text.lower():
        # Remove "draw" keyword (case-insensitive) and any extra whitespace.
        refined_prompt = re.sub(r'\bdraw\b', '', user_text, flags=re.IGNORECASE).strip()
        if user_image is not None:
            # If an image is provided, use image-to-image transformation.
            return generate_img2img(prompt=refined_prompt, init_image=user_image)
        else:
            # Otherwise, generate image from text prompt.
            return generate_text2img(prompt=refined_prompt)
    # If "draw" is not in the prompt, do not run Stable Diffusion.
    return None


# -----------------------------------------------------
# Now we import additional libraries for other modalities:
# Gradio for UI, NumPy for array handling, various transformers for language and image processing,
# and libraries for audio processing, webcam capture, and HTTP requests.
import gradio as gr
import numpy as np
from transformers import (AutoModelForCausalLM, AutoTokenizer, BlipProcessor,
                          BlipForConditionalGeneration, BitsAndBytesConfig)
import tempfile
import time
from pydub import AudioSegment
import whisper
import soundfile as sf
import queue
import pyttsx3
import cv2  # OpenCV for webcam capture
import sounddevice as sd
import httpx

# -----------------------------------------------------
# Global variables for audio recording.
audio_queue = queue.Queue()  # Queue to buffer audio data
global_stream = None  # Global variable for the audio stream

# -----------------------------------------------------
# Load Whisper model for audio transcription.
# Using CPU for inference.
asr_model = whisper.load_model("small", device="cpu")


# -----------------------------------------------------
# Function to retrieve available SAPI5 voices for text-to-speech.
def get_voice_choices():
    """
    Return a list of available SAPI5 voices in the format 'index: Voice Name'.
    """
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    return [f"{i}: {voice.name}" for i, voice in enumerate(voices)]


# -----------------------------------------------------
# Function to perform text-to-speech using SAPI5 and save output as WAV.
def tts_sapi5_to_wav(text, voice_choice, rate):
    """
    Convert text to speech using SAPI5 with a selected voice and rate, saving the output as a WAV file.

    Args:
        text (str): The text to convert to speech.
        voice_choice (str): The chosen voice (with index and name).
        rate (int): Speech rate.

    Returns:
        temp_filename (str): The temporary filename of the saved WAV file.
    """
    engine = pyttsx3.init('sapi5')
    voices = engine.getProperty('voices')
    # Extract the voice index from the selection string.
    index = int(voice_choice.split(":")[0])
    engine.setProperty('voice', voices[index].id)
    engine.setProperty('rate', int(rate))
    # Create a temporary file to store the WAV output.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
        temp_filename = f.name
    engine.save_to_file(text, temp_filename)
    engine.runAndWait()
    time.sleep(0.5)  # Ensure the file is fully written
    return temp_filename


# -----------------------------------------------------
# Function to optionally run TTS if enabled.
def auto_tts(text, auto_tts_enabled, tts_voice, tts_rate):
    """
    If auto_tts_enabled is True and text is not empty, perform TTS on the text.

    Args:
        text (str): The text to convert.
        auto_tts_enabled (bool): Flag to enable TTS.
        tts_voice (str): Selected TTS voice.
        tts_rate (int): TTS speech rate.

    Returns:
        tuple: (text, wav_file_or_None)
    """
    if auto_tts_enabled and text.strip() != "":
        wav_file = tts_sapi5_to_wav(text, tts_voice, tts_rate)
        return text, wav_file
    return text, None


# -----------------------------------------------------
# Audio recording callback function.
# This function safely puts incoming audio data into the queue.
def safe_callback(indata, frames, time_info, status):
    try:
        audio_queue.put(indata.copy())
    except Exception as e:
        print(f"Callback error caught: {e}")


# -----------------------------------------------------
# Function to toggle audio recording on/off.
def toggle_record(is_recording):
    """
    Start or stop the audio recording based on the current state.

    Args:
        is_recording (bool): Current recording state.

    Returns:
        tuple: (new_state, message or transcription text)
    """
    global global_stream, audio_queue
    fs = 16000  # Sample rate for Whisper transcription

    if not is_recording:
        # Clear any previous audio in the queue.
        while not audio_queue.empty():
            audio_queue.get()
        # Start the input stream with a callback for audio capture.
        global_stream = sd.InputStream(
            samplerate=fs,
            channels=1,
            dtype='float32',
            callback=safe_callback
        )
        global_stream.start()
        return True, "Recording... (press button again to stop)"
    else:
        # Stop and close the audio stream.
        global_stream.stop()
        global_stream.close()
        global_stream = None
        # Retrieve all recorded audio from the queue.
        data_list = []
        while not audio_queue.empty():
            data_list.append(audio_queue.get())
        if data_list:
            # Concatenate all audio chunks.
            recorded_audio = np.concatenate(data_list, axis=0)
            # Write the audio data to a temporary WAV file.
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_filename = temp_file.name
            sf.write(temp_filename, recorded_audio, fs)
            # Load the audio data for transcription.
            audio_data = whisper.load_audio(temp_filename)
            try:
                # Transcribe (and translate) the audio using Whisper.
                result = asr_model.transcribe(audio_data, task="translate")
            except Exception as e:
                result = {"text": f"Whisper Error: {e}"}
            os.remove(temp_filename)  # Remove the temporary file.
            return False, result["text"]
        else:
            return False, "No audio recorded."


# -----------------------------------------------------
# Update the UI button text based on recording state.
def update_record_button(is_recording):
    """
    Update the record toggle button's label.

    Args:
        is_recording (bool): Current recording state.

    Returns:
        gr.update: Updated button value.
    """
    if is_recording:
        return gr.update(value="Stop Recording")
    else:
        return gr.update(value="Record & Transcribe (Toggle)")


# -----------------------------------------------------
# Function to capture an image from the webcam using OpenCV.
def capture_webcam_frame(camera_index):
    """
    Capture a single frame from the selected webcam.

    Args:
        camera_index (int or str): Index of the camera.

    Returns:
        PIL.Image or None: Captured image converted to RGB, or None if capture fails.
    """
    cap = cv2.VideoCapture(int(camera_index))
    ret, frame = cap.read()
    cap.release()
    if not ret or frame is None:
        print(f"Error: Could not capture image from camera {camera_index}.")
        return None
    # Convert from BGR (OpenCV default) to RGB for PIL.
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# -----------------------------------------------------
# Vicuna & BLIP model loading and chat setup.

# Set the model path based on the chosen Vicuna version.

model_path = "C:/models/vicuna-13b-v1.5-16k"

# Load the tokenizer for the Vicuna model.
print("Loading Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# Load the Vicuna model in 4-bit mode to save memory.
print("Loading Vicuna Model in 4-bit mode...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    llm_int8_enable_fp32_cpu_offload=False,
    device_map="auto"
)
print("Vicuna Model Loaded Successfully!")

# Set up quantization configuration for BLIP.
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    low_cpu_mem_usage=True
)

# Load the image captioning model (BLIP) in 4-bit mode.
print("Loading image captioning model (BLIP) in 4-bit mode...")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base",
    quantization_config=quant_config
)
print("Image Captioning Model (BLIP) Loaded in 4-bit mode!")

# Warm-up BLIP with a dummy image to ensure it is ready.
dummy_image = Image.new("RGB", (64, 64), color=(255, 0, 0))
try:
    warmup_input = blip_processor(images=dummy_image, return_tensors="pt").to(model.device)
    warmup_ids = blip_model.generate(**warmup_input)
    warmup_caption = blip_processor.batch_decode(warmup_ids, skip_special_tokens=True)[0].strip()
    print("BLIP warm-up complete. Dummy caption:", warmup_caption)
except Exception as e:
    print("BLIP warm-up failed:", e)

# Initialize conversation history as an empty list.
conversation_history = []


# -----------------------------------------------------
# Define function to handle chat responses.
def chat_response(user_text, image_input, camera_index, temperature, top_p, rep_penalty, max_tokens, num_beams):
    """
    Handle the Vicuna chat logic. If an image is provided, generate a BLIP caption
    and add it to the conversation history.

    Args:
        user_text (str): User input text.
        image_input (PIL.Image or None): Optional image provided by the user.
        camera_index (int): Selected camera index.
        temperature (float): Sampling temperature.
        top_p (float): Top-p (nucleus sampling) parameter.
        rep_penalty (float): Repetition penalty.
        max_tokens (int): Maximum tokens to generate.
        num_beams (int): Beam search parameter.

    Returns:
        str: Latest response from the AI.
    """
    global conversation_history

    # Initialize conversation with system prompts if history is empty.
    if not conversation_history:
        conversation_history.append(("system", "You are an AI friend that is multi-lingual but respond in English."))
        conversation_history.append(("system", "You are funny and very friendly."))
        conversation_history.append(("system",
                                     "For every mathematical expression, wrap display equations in $$...$$ and inline equations in \\(...\\)."))

    # If an image is provided, generate a caption using BLIP.
    if image_input is not None:
        try:
            caption_input = blip_processor(images=image_input, return_tensors="pt").to(model.device)
            generated_ids = blip_model.generate(**caption_input)
            image_caption = blip_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            conversation_history.append(("caption", f"Image caption: {image_caption}"))
        except Exception as e:
            print("Image captioning failed:", e)
            conversation_history.append(("caption", "[Error: Unable to generate caption]"))

    # Append the user input to conversation history.
    conversation_history.append(("user", user_text))

    # Build the prompt from the conversation history.
    prompt = "\n".join(f"{role}: {text}" for role, text in conversation_history) + "\nAI:"

    # Tokenize the prompt and move it to the appropriate device.
    input_ids = tokenizer(prompt, return_tensors="pt", padding=True).input_ids.to(model.device)
    with torch.no_grad():
        # Generate the response using the Vicuna model.
        output = model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=float(rep_penalty),
            num_beams=num_beams
        )
    # Decode the generated tokens.
    full_response = tokenizer.decode(output[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # Extract the latest response by splitting on "AI:".
    latest_response = full_response.split("AI:")[-1].strip() if "AI:" in full_response else full_response.strip()

    # Append the AI response to conversation history.
    conversation_history.append(("ai", latest_response))
    return latest_response


# -----------------------------------------------------
# Function to clear the conversation history.
def clear_history():
    """
    Clear the conversation history.

    Returns:
        str: Confirmation message.
    """
    global conversation_history
    conversation_history = []
    return "Conversation history cleared."


# -----------------------------------------------------
# Main logic to handle either chat or stable diffusion.
def record_transcribe_and_submit(transcript, image_input, camera_index, beams, tts_auto, tts_voice):
    """
    Process the transcript and image input to determine whether to run
    Stable Diffusion or normal Vicuna chat.

    Workflow:
    1) If transcript contains 'draw', run Stable Diffusion:
       - Use img2img if an image is present, otherwise text2img.
    2) Otherwise, run the Vicuna chat logic.

    Args:
        transcript (str): User's transcript or text input.
        image_input (PIL.Image or None): Optional image input.
        camera_index (int): Selected camera index.
        beams (int): Beam search value.
        tts_auto (bool): Flag for auto text-to-speech.
        tts_voice (str): Selected TTS voice.

    Returns:
        tuple: (chat_output, transcript, tts_wav, sd_image)
    """
    # Default parameters for Vicuna chat.
    temperature = 0.9
    top_p = 0.9
    rep_penalty = 1.1
    max_tokens = 4096
    tts_rate = 200

    # If the transcript is just the recording indicator, do nothing further.
    if transcript.strip().lower() == "recording... (press button again to stop)":
        return "", transcript, None, None

    # Step 1: Check if the prompt should trigger Stable Diffusion.
    sd_image = maybe_run_stable_diffusion(transcript, image_input)
    if sd_image is not None:
        # If an SD image is generated, skip the chat logic.
        text_out = ""  # No text response from chat.
        tts_wav = None
        return text_out, transcript, tts_wav, sd_image

    # Step 2: Otherwise, process with normal Vicuna chat.
    response = chat_response(
        transcript, image_input, camera_index, temperature, top_p, rep_penalty, max_tokens, beams
    )
    # Optionally convert the response to speech.
    text_out, tts_wav = auto_tts(response, tts_auto, tts_voice, tts_rate)

    return text_out, transcript, tts_wav, None


# -----------------------------------------------------
# Function to process text submitted via the UI (text input submit button).
def process_text(user_text, image_input, camera_index, beams, tts_auto, tts_voice):
    """
    Process the text input and determine whether to run Stable Diffusion
    or Vicuna chat.

    Args:
        user_text (str): User's input text.
        image_input (PIL.Image or None): Optional image input.
        camera_index (int): Selected camera index.
        beams (int): Beam search parameter.
        tts_auto (bool): Flag for auto TTS.
        tts_voice (str): Selected TTS voice.

    Returns:
        tuple: (chat_output, tts_wav, sd_image) where sd_image is None if not generated.
    """
    # Check if the "draw" keyword is present to trigger Stable Diffusion.
    sd_image = maybe_run_stable_diffusion(user_text, image_input)
    if sd_image is not None:
        # If SD is triggered, return only the image.
        return "", None, sd_image

    # Default parameters for chat.
    temperature = 0.9
    top_p = 0.9
    rep_penalty = 1.1
    max_tokens = 50000
    tts_rate = 200

    # Process the text with the chat_response function.
    chat_resp = chat_response(
        user_text, image_input, camera_index, temperature, top_p, rep_penalty, max_tokens, beams
    )
    # Optionally perform text-to-speech.
    text_out, tts_wav = auto_tts(chat_resp, tts_auto, tts_voice, tts_rate)
    return text_out, tts_wav, None


# -----------------------------------------------------
# Gradio UI Setup
# Define custom CSS to style the Gradio interface.
css = """
textarea, input, .gradio-container, .gradio-output, .input_textbox, .output_textbox {
    font-size: 24px !important;
}
.half-width {
    width: 50% !important;
    float: left;
}
"""

# Create the Gradio Blocks interface.
with gr.Blocks(css=css) as demo:
    # Title markdown for the interface.
    gr.Markdown("<h1 align='center'>Multi-Modal Chat w/ Vicuna, BLIP, & 2-Mode Stable Diffusion</h1>")

    # Define row for text input and submit button.
    with gr.Row():
        with gr.Column(scale=3):
            text_input = gr.Textbox(
                label="Enter your query:",
                placeholder="Type your query here...",
                lines=3,
                interactive=True
            )
            submit_button = gr.Button("Submit")
        with gr.Column(scale=1):
            # Allow user to upload or capture an image.
            image_input = gr.Image(label="Upload or Capture Image", type="pil")

    # Row for camera selection and image capture.
    with gr.Row():
        with gr.Column(scale=1):
            camera_selector = gr.Dropdown(
                choices=[0, 1, 2],
                label="Select Camera",
                value=0,
                type="value"
            )
            capture_button = gr.Button("Capture Image from Camera")

    # Row for recording audio.
    with gr.Row():
        record_toggle_button = gr.Button("Record & Transcribe (Toggle)")
        is_recording = gr.State(False)  # Keep track of recording state.

    # Row for TTS options and beam search parameter.
    with gr.Row():
        with gr.Column(scale=1):
            tts_auto = gr.Checkbox(label="Auto TTS", value=True)
            tts_voice = gr.Dropdown(
                choices=get_voice_choices(),
                label="TTS Voice",
                value=get_voice_choices()[0]
            )
        with gr.Column(scale=2):
            beams_dropdown = gr.Dropdown(
                choices=list(range(1, 11)),
                label="Beam Search",
                value=1,
                elem_classes="half-width"
            )

    # Row for displaying chat response and TTS output.
    with gr.Row():
        with gr.Column(scale=3):
            chat_output = gr.Markdown(label="Response")
        with gr.Column(scale=1):
            tts_output = gr.Audio(label="TTS Output (WAV)", type="filepath", autoplay=True)
            audio_input = gr.Audio(label="Upload Audio (WAV, MP3, MP4)", type="filepath")

    # Row for clearing conversation history.
    with gr.Row():
        clear_history_button = gr.Button("Clear History")

    # Row for displaying generated Stable Diffusion image.
    with gr.Row():
        sd_image_output = gr.Image(label="Generated Image from Stable Diffusion", type="pil")


    # -------------------------------------------------
    # Function to process uploaded audio and transcribe it.
    def process_audio(file_path):
        """
        Convert uploaded audio file to text using Whisper.

        Args:
            file_path (str): Path to the uploaded audio file.

        Returns:
            str: Transcribed text.
        """
        if not file_path:
            return ""
        audio_data = whisper.load_audio(file_path)
        try:
            result = asr_model.transcribe(audio_data, task="translate")
        except Exception as e:
            print(f"Error during transcription: {e}")
            return f"Error: {e}"
        return result["text"]


    # When an audio file is uploaded, automatically process it and update the text input.
    audio_input.change(fn=process_audio, inputs=[audio_input], outputs=[text_input])

    # Capture an image from the selected webcam.
    capture_button.click(fn=capture_webcam_frame, inputs=[camera_selector], outputs=image_input)

    # Recording toggle: start recording, update button, and then process the recording.
    record_toggle_button.click(
        fn=toggle_record,
        inputs=[is_recording],
        outputs=[is_recording, text_input]
    ).then(
        fn=update_record_button,
        inputs=[is_recording],
        outputs=[record_toggle_button]
    ).then(
        fn=record_transcribe_and_submit,
        inputs=[text_input, image_input, camera_selector, beams_dropdown, tts_auto, tts_voice],
        outputs=[chat_output, text_input, tts_output, sd_image_output]
    )

    # When the user submits text, process it accordingly.
    text_input.submit(
        fn=process_text,
        inputs=[text_input, image_input, camera_selector, beams_dropdown, tts_auto, tts_voice],
        outputs=[chat_output, tts_output, sd_image_output]
    )
    submit_button.click(
        fn=process_text,
        inputs=[text_input, image_input, camera_selector, beams_dropdown, tts_auto, tts_voice],
        outputs=[chat_output, tts_output, sd_image_output]
    )

    # Clear conversation history when the button is pressed.
    clear_history_button.click(fn=clear_history, outputs=chat_output)

# -----------------------------------------------------
# SSL settings for self-signed certificates (optional).
# Configure environment variables to disable SSL verification and use provided certificates.
os.environ["GRADIO_SSL_NO_VERIFY"] = "1"
os.environ["SSL_CERT_FILE"] = r"C:\Users\tomsp\Downloads\cert.pem"


# Define a custom HTTP transport if needed (currently just inherits default behavior).
class CustomTransport(httpx.HTTPTransport):
    def handle_request(self, request):
        return super().handle_request(request)


# Launch the Gradio interface with SSL configuration.
demo.launch(
    server_name="0.0.0.0",
    server_port=8443,
    ssl_certfile="C:/Users/tomsp/Downloads/cert.pem",
    ssl_keyfile="C:/Users/tomsp/Downloads/key.pem",
    ssl_verify=False,  # Disable SSL verification for self-signed certificates
    share=True
)
