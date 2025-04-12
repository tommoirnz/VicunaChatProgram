Multi-Modal Interactive Assistant with Lighting Control and Image Captioning
Overview
This project is a multi-modal interactive application that integrates speech recognition, text-to-speech (TTS), image captioning via a BLIP module, and lighting control using both a simulated user interface and real Philips Hue lights. The system leverages the Vicuna language model for natural language responses and allows users to control a simulated lighting system with delayed commands.

Features
Speech Recognition & ASR
Users can record their voice to obtain transcriptions.

Text-to-Speech (TTS)
Supports Microsoft Edge TTS and SAPI TTS, allowing spoken playback of responses.

Vicuna Chat Interface
Uses the Vicuna language model for generating contextual responses and command sequences.

Simulated Lighting Control
Provides a graphical user interface (GUI) with a simulation of four lights that can be toggled on/off.

Philips Hue Integration
Controls real Philips Hue lights (for example, lights 1 & 2) based on commands.

Timer Commands
Supports delayed execution of lighting commands using timer commands.

Image Captioning with BLIP
Processes images (via drag-and-drop or camera capture) using the BLIP module. Captions are prefixed with a BLIP: tag so that Vicuna can interpret the image description and respond accordingly.

Clipboard Paste Option
A “Paste” button in the Vicuna chat window to quickly paste text from the clipboard.

Requirements
Python 3.8+

Required Python libraries (install via pip):

tkinter (usually built-in)
numpy
sounddevice
speech_recognition
edge_tts
pydub
simpleaudio
matplotlib
Pillow
torch
transformers
langdetect
pyttsx3
opencv-python
phue
tkinterdnd2 (optional for drag-and-drop functionality)
A Vicuna model (e.g., vicuna-13b-v1.5-16k) downloaded locally.

Philips Hue Bridge and compatible Philips Hue lights (for physical integration).

Installation & Setup
Clone the Repository

git clone https://github.com/yourusername/interactive-assistant.git
cd interactive-assistant
Install Dependencies
Use pip to install all necessary libraries:

pip install numpy sounddevice speechrecognition edge-tts pydub simpleaudio matplotlib pillow torch transformers langdetect pyttsx3 opencv-python phue tkinterdnd2
Configure Environment Variables
Make sure the environment variable for tkdnd files is set. For example, in your code (or as a system variable):

os.environ["TKDND_LIBRARY"] = r"C:\tkdnd"
Configure the Philips Hue Bridge IP
In the main execution block of the program, update the BRIDGE_IP variable to match your Hue Bridge’s IP address.

Download and Set Up the Vicuna Model
Make sure that the path in the vicuna_model_path variable points to your locally stored Vicuna model directory.

Usage
Run the program by executing the main Python file:

python your_program.py
This will launch several windows:

Combined ASR Application Window for recording and transcribing speech.
Transcript Window to view the transcription.
Vicuna Chat Window for natural language interaction and command submission.
Image Drop Window for drag-and-drop or camera-captured image captioning.
Light Simulator Window displaying the simulated lights.
Lighting Commands
Basic Commands
Turn a Single Light On/Off
For instance:

command light 1 on
command light 3 off
Turn All Lights On/Off
For example:

command all lights on
command all lights off
Timer (Delayed) Commands
You can schedule delays between commands with timer commands. The format is as follows:

Syntax:
COMMAND timer X
where X is the delay in seconds.

Example:
To turn light 1 on after 30 seconds and then turn it off 10 seconds later (cumulative delay), use:

command timer 30 command light 1 on command timer 10 command light 1 off
Note: The system uses a cumulative delay; if no explicit initial delay is specified, it will not add one.

Natural Language Instructions Example
When you say something like:

turn light one on for 10 seconds and then turn it off again
Vicuna (based on the system prompt) should reply with:

COMMAND light 1 on COMMAND timer 10 COMMAND light 1 off
Be careful: If an extra timer is added at the start, adjust your instructions in the system prompt to prevent an initial delay if not intended.

Image Captioning with BLIP
When using the Image Drop Window, you can either:

Drag and drop an image file.
Click to load an image.
Capture an image from a selected camera.
The BLIP module processes the image and generates a caption.

The output is automatically tagged with BLIP: (for example, BLIP: a dog playing in the park on a sunny day).

Vicuna interprets this tag according to the system prompt and responds with commentary or clarifying questions about the image.

Customizing the System Prompt
The system prompt (in the vicuna_chat_response function) is key to guiding Vicuna’s behavior. It specifies:

How Vicuna should generate lighting commands.
How to format timer commands.
Instructions on handling BLIP image captions (look for messages starting with BLIP:).
You can modify this prompt as needed to adjust the responses.

Troubleshooting
Timer Not Working:
If you do not see the expected delay when using timer commands, ensure that:

The command string is formatted properly, with the word "timer" included.
You are not using the direct command branch (i.e. commands with "timer" should be processed via the timer-enabled parser).
Response Discrepancies:
The console might print more detailed (debug) outputs than what appears in the Vicuna chat window due to preprocessing and debug print statements. Remove or adjust debug prints as desired.

TTS Issues:
If TTS stops working during timer commands, verify that the timer-enabled commands do not inadvertently trigger TTS (TTS is only triggered when the response is not a command).

Contributions
Contributions, improvements, and bug reports are welcome. Please open an issue or submit a pull request if you have suggestions.

License
This project is released under the MIT License.
