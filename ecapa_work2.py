from fastapi import FastAPI, WebSocket
import asyncio
import sounddevice as sd
import numpy as np
import whisper
from scipy.spatial.distance import cosine
from speechbrain.inference.speaker import EncoderClassifier
import torchaudio
import scipy.io.wavfile as wav
import re
from torchaudio.transforms import Resample
import torch
from openai import OpenAI

# Константы
RATE = 44100
CHUNK = 1024
BUFFER_SEC = 2
ADMIN_AUDIO_PATH = r"C:\Users\semen\Downloads\диплом\admin_voice_data\admin_reference.wav"

# Ресэмплер
resampler = Resample(orig_freq=RATE, new_freq=16000)

# LLM клиент
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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"
)

# Карта команд
COMMAND_MAP = {
    "поворачивай налево": "turn_left",
    "поворачивай направо": "turn_right",
    "стоп": "stop",
    "вперед": "start"
}

# Инициализация моделей
whisper_model = whisper.load_model("base")
classifier = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": "cpu"}
)


def load_audio_embedding(audio_path: str) -> np.ndarray:
    signal, _ = torchaudio.load(audio_path)
    if signal.shape[0] > 1:
        signal = signal.mean(dim=0, keepdim=True)
    embeddings = classifier.encode_batch(signal)
    return np.squeeze(embeddings[0].cpu().detach().numpy())


def extract_text_from_audio(audio_bytes: bytes) -> str:
    audio_array = np.frombuffer(audio_bytes, np.int16).astype(np.float32) / 32768.0
    waveform = torch.from_numpy(audio_array).unsqueeze(0)
    waveform_16k = resampler(waveform).squeeze(0).numpy()
    result = whisper_model.transcribe(audio=waveform_16k, language="ru")
    return result.get("text", "").strip()


async def classify_command(text: str) -> str:
    key = text.lower().strip()
    if key in COMMAND_MAP:
        return COMMAND_MAP[key]
    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"Выбери ближайшую команду к: turn_left, turn_right, stop, start. Текст: {text}"}
        ]
    )
    try:
        return resp.choices[0].message.content.strip()
    except Exception:
        return "unknown"


def compare_embeddings(reference: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> bool:
    return cosine(reference, target) < threshold


def save_temp_wav(filename: str, audio_data: bytes) -> None:
    audio_np = np.frombuffer(audio_data, np.int16).reshape(-1, 1)
    wav.write(filename, RATE, audio_np)


def split_sentences(text: str) -> list:
    return re.split(r'(?<=[\.\!\?])\s+', text)


# Загрузка эталонного embedding
admin_emb = load_audio_embedding(ADMIN_AUDIO_PATH)

# FastAPI
app = FastAPI()


@app.websocket("/ws/audio")
async def audio_websocket(ws: WebSocket):
    await ws.accept()
    buf = bytearray()
    stream = sd.InputStream(samplerate=RATE, channels=1, dtype='int16', blocksize=CHUNK)
    stream.start()

    try:
        while True:
            data, _ = stream.read(CHUNK)
            buf.extend(data.tobytes())

            if len(buf) >= RATE * 2 * BUFFER_SEC:
                audio_chunk = bytes(buf[:RATE * 2 * BUFFER_SEC])
                del buf[:RATE * 2 * BUFFER_SEC]

                temp_path = "temp.wav"
                save_temp_wav(temp_path, audio_chunk)
                user_emb = load_audio_embedding(temp_path)
                is_admin = compare_embeddings(admin_emb, user_emb)

                text = extract_text_from_audio(audio_chunk)
                for sentence in split_sentences(text):
                    if sentence:
                        command = await classify_command(sentence)
                        await ws.send_json({
                            "is_admin": is_admin,
                            "command": command
                        })

            await asyncio.sleep(0.01)

    finally:
        stream.stop()
        stream.close()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=4959)
