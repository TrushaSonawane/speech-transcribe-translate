# pyright: reportAttributeAccessIssue=false
"""
GUI: Record → Transcribe (offline Vosk) → Translate (Argos offline → deep-translator online)
- Preloads Vosk model once (background)
- Streams audio to recognizer (no temp WAV)
- Microphone device picker
- Auto-stop on silence
- Shows transcript and translation

Run:
  python gui_transcribe_translate.py
"""

import os
import sys
import json
import math
import queue
import threading
import socket
from typing import Optional, Tuple, List

import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

import customtkinter as ctk
from tkinter import messagebox


# ===================== CONFIG =====================
MODEL_PATH = "model"       # folder containing your Vosk model
SAMPLE_RATE = 16_000
CHANNELS = 1

# Voice-activity / silence detection
FRAME_SECONDS = 0.30             # block length for analysis
SILENCE_RMS_THRESHOLD = 0.010    # lower => more sensitive to quiet speech (0.008–0.015 typical)
SILENCE_HOLD_SECONDS = 1.8       # continuous silence to stop
MIN_RECORD_SECONDS = 1.5         # don't stop before this much audio

# UI
DEFAULT_SOURCE_LANG = "en"       # spoken language code (used for Argos direction)
DEFAULT_TARGET_LANG = "fr"       # translate into
THEME = "System"                 # "System" / "Dark" / "Light"
ACCENT_THEME = "blue"            # built-in: "blue" / "dark-blue" / "green"
# ==================================================


# -------- Translation helpers --------
try:
    from deep_translator import GoogleTranslator as _GoogleTranslator
    HAS_DEEP = True
except Exception:
    _GoogleTranslator = None
    HAS_DEEP = False


def argos_translate(text: str, src: str, tgt: str) -> Tuple[Optional[str], Optional[str]]:
    """Offline translation if Argos language pair is installed."""
    try:
        import argostranslate.translate as A
    except Exception:
        return None, "Argos not installed"
    langs = A.get_installed_languages()
    src_lang = next((l for l in langs if getattr(l, "code", None) == src), None)
    tgt_lang = next((l for l in langs if getattr(l, "code", None) == tgt), None)
    if not src_lang or not tgt_lang:
        return None, f"Argos languages not installed: src={bool(src_lang)}, tgt={bool(tgt_lang)}"
    try:
        translator = src_lang.get_translation(tgt_lang)
    except Exception as e:
        return None, f"Argos get_translation failed: {e}"
    if translator is None or not hasattr(translator, "translate"):
        return None, "Argos pair not installed for this direction."
    try:
        return str(translator.translate(text)), None
    except Exception as e:
        return None, f"Argos translation failed: {e}"


def online_translate(text: str, tgt: str) -> Tuple[Optional[str], Optional[str]]:
    """Online translation using deep-translator (Google)."""
    if not HAS_DEEP or _GoogleTranslator is None:
        return None, "deep-translator not installed"
    try:
        return _GoogleTranslator(source="auto", target=tgt).translate(text), None
    except Exception as e:
        return None, f"Online translation failed: {e}"


def internet_available(host: str = "8.8.8.8", port: int = 53, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except Exception:
        return False


# -------- Audio helpers --------
def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32, copy=False)
    if x.ndim == 2:
        x = np.mean(x, axis=1)
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def float_to_int16(block: np.ndarray) -> bytes:
    """Convert float32 [-1,1] to int16 PCM bytes (mono if needed)."""
    y = np.clip(block, -1.0, 1.0)
    y = (y * 32767.0).astype(np.int16)
    if y.ndim == 2 and y.shape[1] > 1:
        y = np.mean(y, axis=1).astype(np.int16)
    return y.tobytes()


# -------- Device listing (module-level, Pylance-safe) --------
def list_input_devices() -> Tuple[List[str], List[int]]:
    names: List[str] = []
    ids: List[int] = []
    try:
        dev_list = sd.query_devices()
        for idx in range(len(dev_list)):       # iterate by index (stubs-safe)
            info = sd.query_devices(idx)       # returns a dict for that index
            max_in = info.get("max_input_channels", 0)
            if isinstance(max_in, (int, float)) and max_in > 0:
                dev_name = info.get("name", f"Device {idx}")
                names.append(f"{idx}: {dev_name}")
                ids.append(idx)
    except Exception:
        pass
    if not names:
        return (["Default"], [-1])
    return names, ids


# ===================== GUI APP =====================
class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        ctk.set_appearance_mode(THEME)
        ctk.set_default_color_theme(ACCENT_THEME)

        self.title("Speech → Transcript → Translation")
        self.geometry("980x680")
        self.minsize(920, 640)

        # state
        self.stop_event = threading.Event()
        self.audio_q: queue.Queue = queue.Queue()
        self.worker: Optional[threading.Thread] = None
        self.model: Optional[Model] = None
        self.model_ready = threading.Event()

        # ===== top bar =====
        top = ctk.CTkFrame(self, corner_radius=12, fg_color=("gray95", "gray10"))
        top.pack(fill="x", padx=14, pady=12)

        self.src_lang = ctk.StringVar(value=DEFAULT_SOURCE_LANG)
        self.tgt_lang = ctk.StringVar(value=DEFAULT_TARGET_LANG)

        langs = [
            "en", "en-IN", "fr", "es", "de", "it", "pt",
            "hi", "bn", "ta", "te", "tr", "ru", "ja", "ko", "zh-CN", "ar", "ur", "sv"
        ]
        ctk.CTkLabel(top, text="Speak (ASR/translation source):").grid(row=0, column=0, padx=10, pady=8, sticky="w")
        self.src_combo = ctk.CTkComboBox(top, values=langs, variable=self.src_lang, width=140)
        self.src_combo.grid(row=0, column=1, padx=10, pady=8, sticky="w")

        ctk.CTkLabel(top, text="Translate to:").grid(row=0, column=2, padx=10, pady=8, sticky="w")
        self.tgt_combo = ctk.CTkComboBox(top, values=langs, variable=self.tgt_lang, width=140)
        self.tgt_combo.grid(row=0, column=3, padx=10, pady=8, sticky="w")

        # input device picker
        ctk.CTkLabel(top, text="Input device:").grid(row=0, column=4, padx=(20, 6), pady=8, sticky="e")
        self.device_names, self.device_ids = list_input_devices()
        self.device_var = ctk.StringVar(value=self.device_names[0] if self.device_names else "Default")
        self.device_combo = ctk.CTkComboBox(top, values=self.device_names or ["Default"], variable=self.device_var, width=260)
        self.device_combo.grid(row=0, column=5, padx=6, pady=8, sticky="w")

        # buttons
        self.record_btn = ctk.CTkButton(top, text="● Start Recording", command=self.start_recording, width=160, state="disabled")
        self.record_btn.grid(row=0, column=6, padx=(18, 10), pady=8)

        self.stop_btn = ctk.CTkButton(top, text="■ Stop", command=self.stop_recording, width=100, state="disabled")
        self.stop_btn.grid(row=0, column=7, padx=6, pady=8)

        # status
        self.status = ctk.CTkLabel(top, text="Loading speech model…", text_color=("black", "white"))
        self.status.grid(row=1, column=0, columnspan=8, padx=10, pady=(0, 10), sticky="w")

        # ===== text panes =====
        mid = ctk.CTkFrame(self, corner_radius=12)
        mid.pack(fill="both", expand=True, padx=14, pady=(0, 12))

        left = ctk.CTkFrame(mid, corner_radius=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=12)
        right = ctk.CTkFrame(mid, corner_radius=12)
        right.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=12)

        ctk.CTkLabel(left, text="🧾 Transcript").pack(anchor="w", padx=12, pady=(12, 6))
        self.transcript_box = ctk.CTkTextbox(left, height=10)
        self.transcript_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        ctk.CTkLabel(right, text="🌍 Translation").pack(anchor="w", padx=12, pady=(12, 6))
        self.translation_box = ctk.CTkTextbox(right, height=10)
        self.translation_box.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # preload model in background
        threading.Thread(target=self._load_model, daemon=True).start()

    def _load_model(self) -> None:
        try:
            if not os.path.isdir(MODEL_PATH):
                raise FileNotFoundError(
                    f"Vosk model folder not found at:\n{os.path.abspath(MODEL_PATH)}\n"
                    "Download a model (e.g., vosk-model-en-us-0.22) and set MODEL_PATH."
                )
            self.model = Model(MODEL_PATH)
            self.model_ready.set()
            self.set_status("Model ready. Click Start Recording.")
            self.record_btn.configure(state="normal")
        except Exception as e:
            messagebox.showerror("Model load error", str(e))
            self.set_status("Model load failed.")
            self.record_btn.configure(state="disabled")

    def set_status(self, text: str) -> None:
        self.status.configure(text=text)
        self.update_idletasks()

    # --- recording / pipeline ---
    def start_recording(self) -> None:
        if not self.model_ready.is_set():
            self.set_status("Model still loading…")
            return
        if self.worker and self.worker.is_alive():
            return

        # clear state
        with self.audio_q.mutex:
            self.audio_q.queue.clear()
        self.stop_event.clear()
        self.transcript_box.delete("1.0", "end")
        self.translation_box.delete("1.0", "end")

        # pick device id
        pick = self.device_var.get()
        try:
            device_id = int(pick.split(":")[0])
        except Exception:
            device_id = None  # default

        self.record_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.set_status("Listening… auto-stops after brief silence.")

        self.worker = threading.Thread(target=self._run_pipeline, args=(device_id,), daemon=True)
        self.worker.start()

    def stop_recording(self) -> None:
        self.stop_event.set()
        self.set_status("Stopping…")

    def _run_pipeline(self, device_id: Optional[int]) -> None:
        try:
            assert self.model is not None
            rec = KaldiRecognizer(self.model, SAMPLE_RATE)
            rec.SetWords(True)

            blocksize = int(SAMPLE_RATE * FRAME_SECONDS)
            silence_frames_required = int(math.ceil(SILENCE_HOLD_SECONDS / FRAME_SECONDS))
            consecutive_silence = 0
            blocks_seen = 0

            def _cb(indata, frames, time_info, status):
                if status:
                    print(status, file=sys.stderr)
                if not self.stop_event.is_set():
                    self.audio_q.put(indata.copy())

            with sd.InputStream(samplerate=SAMPLE_RATE,
                                blocksize=blocksize,
                                channels=CHANNELS,
                                dtype='float32',
                                device=device_id,
                                callback=_cb):
                while not self.stop_event.is_set():
                    try:
                        data = self.audio_q.get(timeout=0.1)
                    except queue.Empty:
                        continue

                    blocks_seen += 1
                    rec.AcceptWaveform(float_to_int16(data))

                    level = rms(data)
                    if level < SILENCE_RMS_THRESHOLD:
                        consecutive_silence += 1
                    else:
                        consecutive_silence = 0

                    total_seconds = blocks_seen * FRAME_SECONDS
                    if consecutive_silence >= silence_frames_required and total_seconds >= MIN_RECORD_SECONDS:
                        break

            final = json.loads(rec.FinalResult())
            text = (final.get("text") or "").strip()
            if text:
                if text[0].islower():
                    text = text[0].upper() + text[1:]
                if text[-1] not in ".!?":
                    text += "."
            else:
                self.transcript_box.insert("end", "[No speech recognized]")
                self.set_status("Ready.")
                return

            self.transcript_box.insert("end", text)
            self.transcript_box.see("end")
            self.set_status("Transcription done. Translating…")

            # translate
            src = self.src_lang.get().split("-")[0] or "en"
            tgt = self.tgt_lang.get().split("-")[0] or "fr"

            translated: Optional[str] = None
            msg: Optional[str] = None

            translated, msg = argos_translate(text, src, tgt)
            used = "Argos (offline)" if translated else None

            if translated is None and internet_available():
                t2, msg2 = online_translate(text, tgt)
                if t2 is not None:
                    translated = t2
                    used = "Google (online)"
                else:
                    msg = msg2

            if translated:
                self.translation_box.insert("end", translated)
                self.translation_box.see("end")
                self.set_status(f"Done. Translation via {used}.")
            else:
                self.translation_box.insert("end", f"[Translation unavailable: {msg}]")
                self.translation_box.see("end")
                self.set_status("Ready.")

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.set_status("Error.")
        finally:
            with self.audio_q.mutex:
                self.audio_q.queue.clear()
            self.stop_event.clear()
            self.record_btn.configure(state="normal")
            self.stop_btn.configure(state="disabled")


def main() -> None:
    ctk.set_appearance_mode(THEME)
    ctk.set_default_color_theme(ACCENT_THEME)
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
