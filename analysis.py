import os
import sounddevice as sd
import numpy as np
import tempfile
import wave
import csv
import datetime
import time
import re
from groq import Groq

# ✅ Google Sheets
import gspread
from google.oauth2.service_account import Credentials

# ✅ Initialize Groq client
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# 🎤 Audio settings
SAMPLE_RATE = 16000
CHANNELS = 1
MIN_DURATION = 30    # minimum recording time (seconds)
MAX_DURATION = 180   # maximum recording time (3 minutes)
csv_file = "groq_transcripts.csv"

print("🎤 Groq Speech-to-Text + Sentiment + Emotion")
print(f"🎙 Recording for at least {MIN_DURATION} sec (up to {MAX_DURATION} sec)...")

# 🔹 Step 1: Calibrate silence
print("\n🤫 Calibrating... Please stay silent for 3 seconds...")
calibration = sd.rec(
    int(3 * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32'
)
sd.wait()

baseline = np.linalg.norm(calibration) / len(calibration)
SILENCE_THRESHOLD = baseline * 1.5
print(f"✅ Calibration done. Silence baseline={baseline:.4f}, threshold={SILENCE_THRESHOLD:.4f}")

# 🔹 Step 2: Record
print("\n🎙 Recording started... Speak now!")
recording = sd.rec(
    int(MAX_DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype='float32'
)
start_time = time.time()

while True:
    elapsed = time.time() - start_time
    if elapsed >= MIN_DURATION:
        # Check silence
        volume_norm = np.linalg.norm(recording[:int(elapsed * SAMPLE_RATE)]) / (elapsed * SAMPLE_RATE)
        if volume_norm < SILENCE_THRESHOLD or elapsed >= MAX_DURATION:
            break
    time.sleep(1)

sd.stop()
print(f"🛑 Recording stopped at {elapsed:.1f} seconds.")

# 🔹 Step 3: Trim trailing silence
energy = np.abs(recording.flatten())
silence_cutoff = 0.001  # adjust if needed
if np.any(energy > silence_cutoff):
    last_non_silent = np.max(np.where(energy > silence_cutoff))
    trimmed_recording = recording[:last_non_silent + 1]
else:
    trimmed_recording = recording

# 🔹 Step 4: Save to temp WAV file
tmp_filename = None
with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
    tmp_filename = tmp_file.name
    with wave.open(tmp_file.name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit PCM
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes((trimmed_recording * 32767).astype(np.int16).tobytes())

# 🔹 Step 5: Transcribe with Groq Whisper
text, sentiment_result, emotion_result = "Not Speaking", "N/A", "N/A"
with open(tmp_filename, "rb") as f:
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=f,
        temperature=0  # deterministic, fewer repetitions
    )
text = transcription.text.strip()

# 🔹 Clean repeated words like "you you you"
def clean_repetitions(text):
    return re.sub(r'\b(\w+)( \1){2,}\b', r'\1', text)

text = clean_repetitions(text)

if text:
    print(f"📝 Transcript: {text}")

    # 📊 Sentiment
    sentiment = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Classify sentiment as one word: Positive, Negative, or Neutral."},
            {"role": "user", "content": text}
        ]
    )
    sentiment_result = sentiment.choices[0].message.content.strip()

    # 🎭 Emotion
    emotion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": "Detect emotion as one word: Joy, Sadness, Anger, Fear, or Surprise."},
            {"role": "user", "content": text}
        ]
    )
    emotion_result = emotion.choices[0].message.content.strip()

# 🔹 Step 6: Save results to CSV
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
file_exists = os.path.isfile(csv_file)
with open(csv_file, "a", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    if not file_exists:
        writer.writerow(["Timestamp", "Transcript", "Sentiment", "Emotion"])
    writer.writerow([timestamp, text, sentiment_result, emotion_result])

print(f"📊 Sentiment: {sentiment_result} | 🎭 Emotion: {emotion_result}")
print(f"✅ Saved to {csv_file}")

# 🔹 Step 7: Save to Google Sheets
try:
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
    SERVICE_ACCOUNT_FILE = "credentials.json"  # make sure this file exists in same folder

    creds = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    client_gs = gspread.authorize(creds)

    # Open your Google Sheet
    sheet = client_gs.open_by_url("https://docs.google.com/spreadsheets/d/12BSvCzKWccU8DlhANNbW-94p1D73rH4Lkjojhg-URfo/edit?usp=sharing")
    worksheet = sheet.sheet1

    # Append row
    worksheet.append_row([timestamp, text, sentiment_result, emotion_result])
    print("✅ Data also saved to Google Sheet.")
except Exception as e:
    print(f"⚠ Could not save to Google Sheets: {e}")

# Cleanup
if tmp_filename and os.path.exists(tmp_filename):
    try:
        os.remove(tmp_filename)
    except PermissionError:
        print("⚠ Temp file cleanup skipped (file in use).")
