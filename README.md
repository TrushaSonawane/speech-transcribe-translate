🎙️ Speech-Transcribe-Translate

Speech → Transcription → Translation System
Built with Python, Vosk, Argos Translate, and Deep Translator

🧠 Overview

Speech-Transcribe-Translate is a Python-based desktop application that captures audio through your microphone, transcribes it in real time, and translates it into a target language.

It’s designed with offline resilience and hybrid intelligence in mind — using Vosk and Argos Translate for completely offline operation, while automatically switching to Deep Translator when internet connectivity is available.

This project was built to explore how speech systems can remain reliable, private, and performant even without cloud dependencies.

✨ Key Features

🎧 Real-time speech transcription (offline using Vosk
)

🌐 Hybrid translation system — offline via Argos Translate
, online via Deep Translator

🧩 Automatic language fallback between offline and online translation

🕓 Silence-based start/stop detection (no need to manually end recording)

💬 Instant transcript and translation display in a clean CustomTkinter GUI

🔒 Privacy-friendly — runs entirely on your local machine when offline

🛠️ Tech Stack
Category	Technology
Speech Recognition	Vosk

Translation	Argos Translate
, Deep Translator

UI Framework	CustomTkinter

Audio Processing	sounddevice, numpy
Speech Output (optional)	gtts, playsound
⚙️ Installation

Clone the repository:

git clone https://github.com/<your-username>/speech-transcribe-translate.git
cd speech-transcribe-translate


Install dependencies:

pip install -r requirements.txt


If you don’t have a requirements.txt, you can install manually:

pip install vosk argostranslate deep-translator sounddevice numpy customtkinter gtts playsound

🧩 Model Setup (Vosk)

This app uses a Vosk ASR model for offline speech recognition.
You need to download one model for your chosen language.

1️⃣ Pick the right model

Recommended (default):

vosk-model-en-us-0.22 → Best accuracy for US/Canadian English

Indian English (better for local accent):

vosk-model-en-in-0.5

Lightweight / Fast (less accurate):

vosk-model-small-en-us-0.15

📦 Download models from the Vosk Models Page
.

2️⃣ Place the model

Unzip the model and rename the folder to model, then put it inside your project directory:

speech-transcribe-translate/
├─ gui_transcribe_translate.py
├─ hybrid_transcribe_translate.py
└─ model/
   ├─ am/
   ├─ conf/
   ├─ graph/
   └─ ...


Add this line to .gitignore:

model/
*.zip

3️⃣ Model Path

The app defaults to:

MODEL_PATH = "model"


If your model is elsewhere, update that variable near the top of the script.

4️⃣ Verify the model

Run:

python gui_transcribe_translate.py


✅ If the model is loaded correctly, the GUI will show “Model ready. Click Start Recording.”

▶️ Running the App

To launch the GUI:

python gui_transcribe_translate.py


To use the CLI/Hybrid version:

python hybrid_transcribe_translate.py
